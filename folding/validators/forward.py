import os
import time
import torch
import pickle
import bittensor as bt
from typing import List, Tuple, Dict

from folding.validators.protein import Protein
from folding.utils.uids import get_random_uids
from folding.utils.logging import log_event
from folding.validators.reward import get_rewards
from folding.protocol import FoldingSynapse

from folding.utils.ops import select_random_pdb_id
from folding.validators.hyperparameters import HyperParameters

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDB_PATH = os.path.join(ROOT_DIR, "pdb_ids.pkl")
if not os.path.exists(PDB_PATH):
    raise ValueError(
        f"Required Pdb file {PDB_PATH!r} was not found. Run `python scripts/gather_pdbs.py` first."
    )

with open(PDB_PATH, "rb") as f:
    PDB_IDS = pickle.load(f)


async def run_step(
    self,
    protein: Protein,
    k: int,
    timeout: float,
    exclude: list = [],
):
    bt.logging.debug("run_step")
    start_time = time.time()

    # Get the list of uids to query for this step.
    uids = get_random_uids(self, k=k, exclude=exclude).to(self.device)
    axons = [self.metagraph.axons[uid] for uid in uids]
    synapse = FoldingSynapse(pdb_id=protein.pdb_id, md_inputs=protein.md_inputs)

    # Make calls to the network with the prompt.
    responses: List[FoldingSynapse] = await self.dendrite(
        axons=axons,
        synapse=synapse,
        timeout=timeout,
    )
    # Compute the rewards for the responses given the prompt.
    rewards: torch.FloatTensor = get_rewards(protein, responses)

    # # Log the step event.
    event = {
        "block": self.metagraph.block,
        "step_length": time.time() - start_time,
        "uids": uids.tolist(),
        "response_times": [
            resp.dendrite.process_time if resp.dendrite.process_time != None else 0
            for resp in responses
        ],
        "response_status_messages": [
            str(resp.dendrite.status_message) for resp in responses
        ],
        "response_status_codes": [str(resp.dendrite.status_code) for resp in responses],
        # "rewards": rewards.tolist(),
    }

    return event
    # os.system("pm2 stop v1")

    # # Find the best response given the rewards vector.
    # best: str = responses[rewards.argmax(dim=0)]

    # # Compute forward pass rewards, assumes followup_uids and answer_uids are mutually exclusive.
    # # shape: [ metagraph.n ]
    # scattered_rewards: torch.FloatTensor = self.scores.scatter(0, uids, rewards).to(
    #     self.device
    # )

    # # Update moving_averaged_scores with rewards produced by this step.
    # # shape: [ metagraph.n ]
    # alpha: float = self.config.neuron.moving_average_alpha
    # self.scores = alpha * scattered_rewards + (1 - alpha) * self.scores.to(self.device)

    # bt.logging.debug("event:", str(event))
    # if not self.config.neuron.dont_save_events:
    #     logger.log("EVENTS", "events", **event)


def parse_config(config) -> List[str]:
    """
    Parse config to check if key hyperparameters are set.
    If they are, exclude them from hyperparameter search.
    """
    ff = config.ff
    water = config.water
    box = config.box
    exclude_in_hp_search = []

    if ff is not None:
        exclude_in_hp_search.append("FF")
    if water is not None:
        exclude_in_hp_search.append("WATER")
    if box is not None:
        exclude_in_hp_search.append("BOX")

    return exclude_in_hp_search


async def forward(self):
    while True:
        forward_start_time = time.time()
        exclude_in_hp_search = parse_config(self.config)

        # We need to select a random pdb_id outside of the protein class.
        pdb_id = (
            select_random_pdb_id(PDB_IDS=PDB_IDS)
            if self.config.pdb_id is None
            else self.config.pdb_id
        )
        hp_sampler = HyperParameters(exclude=exclude_in_hp_search)

        for iteration_num in range(hp_sampler.TOTAL_COMBINATIONS):
            hp_sampler_time = time.time()

            event = {}
            try:
                sampled_combination: Dict = hp_sampler.sample_hyperparameters()
                bt.logging.info(
                    f"pdb_id: {pdb_id}, Selected hyperparameters: {sampled_combination}, iteration {iteration_num}"
                )

                protein = Protein(
                    pdb_id=self.config.protein.pdb_id,
                    ff=self.config.protein.ff
                    if self.config.protein.ff is not None
                    else sampled_combination["FF"],
                    water=self.config.protein.water
                    if self.config.protein.water is not None
                    else sampled_combination["WATER"],
                    box=self.config.protein.box
                    if self.config.protein.box is not None
                    else sampled_combination["BOX"],
                    config=self.config.protein,
                )

                hps = {
                    "FF": protein.ff,
                    "WATER": protein.water,
                    "BOX": protein.box,
                    "BOX_DISTANCE": sampled_combination["BOX_DISTANCE"],
                }

                bt.logging.info(f"Attempting to generate challenge: {protein}")
                protein.forward()

            except Exception as E:
                bt.logging.error(
                    f"❌❌ Error running hyperparameters {sampled_combination} for pdb_id {pdb_id} ❌❌"
                )
                bt.logging.warning(E)
                event["status"] = False

            finally:
                event["pdb_id"] = pdb_id
                event.update(hps)  # add the dictionary of hyperparameters to the event
                event["hp_sample_time"] = time.time() - hp_sampler_time

                if "status" not in event:
                    bt.logging.info("✅✅ Simulation ran successfully! ✅✅")
                    event["status"] = True  # simulation passed!
                    break  # break out of the loop if the simulation was successful

                log_event(
                    self, event
                )  # only log the event if the simulation was not successful

        # If we exit the for loop without breaking, it means all hyperparameter combinations failed.
        if event["status"] is False:
            bt.logging.error(
                f"❌❌ All hyperparameter combinations failed for pdb_id {pdb_id}.. Skipping! ❌❌"
            )
            continue  # Skip to the next pdb_id

        # The following code only runs if we have a successful run!
        miner_event = await run_step(
            self,
            protein=protein,
            k=self.config.neuron.sample_size,
            timeout=self.config.neuron.timeout,
        )

        event.update(miner_event)
        event["forward_time"] = time.time() - forward_start_time

        bt.logging.success("✅ Logging pdb results to wandb ✅")
        log_event(self, event)  # Log the entire pipeline.
