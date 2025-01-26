import os
from typing import Dict, Any
from itertools import chain
from collections import defaultdict

import numpy as np
import pandas as pd
from openmm import app

from folding.utils.logger import logger
from folding.base.evaluation import BaseEvaluator
from folding.base.simulation import OpenMMSimulation
from folding.utils import constants as c
from folding.utils.ops import ValidationError, load_pkl, write_pkl, load_pdb_file, save_files, save_pdb, create_velm


class SyntheticMDEvaluator(BaseEvaluator):
    def __init__(self, pdb_id: str, priority: float = 1, **kwargs):
        self.pdb_id = pdb_id
        self.priority = priority
        self.kwargs = kwargs
        self.md_simulator = OpenMMSimulation()

    def process_md_output(
        self, md_output: dict, seed: int, state: str, hotkey: str, basepath: str, pdb_location: str, **kwargs
    ) -> bool:
        self.pdb_location = pdb_location
        required_files_extensions = ["cpt", "log"]
        self.hotkey_alias = hotkey[:8]
        self.current_state = state
        self.miner_seed = seed

        # This is just mapper from the file extension to the name of the file stores in the dict.
        self.md_outputs_exts = {k.split(".")[-1]: k for k, v in md_output.items() if len(v) > 0}

        if len(md_output.keys()) == 0:
            logger.warning(f"Miner {self.hotkey_alias} returned empty md_output... Skipping!")
            return False

        for ext in required_files_extensions:
            if ext not in self.md_outputs_exts:
                logger.error(f"Missing file with extension {ext} in md_output")
                return False

        self.miner_data_directory = os.path.join(basepath, hotkey[:8])

        # Save files so we can check the hash later.
        save_files(
            files=md_output,
            output_directory=self.miner_data_directory,
        )

        try:
            # NOTE: The seed written in the self.system_config is not used here
            # because the miner could have used something different and we want to
            # make sure that we are using the correct seed.

            logger.info(f"Recreating miner {self.hotkey_alias} simulation in state: {self.current_state}")
            self.simulation, self.system_config = self.md_simulator.create_simulation(
                pdb=load_pdb_file(pdb_file=self.pdb_location),
                system_config=self.system_config.get_config(),
                seed=self.miner_seed,
            )

            checkpoint_path = os.path.join(self.miner_data_directory, f"{self.current_state}.cpt")
            state_xml_path = os.path.join(self.miner_data_directory, f"{self.current_state}.xml")
            log_file_path = os.path.join(self.miner_data_directory, self.md_outputs_exts["log"])

            self.simulation.loadCheckpoint(checkpoint_path)

            self.log_file = pd.read_csv(log_file_path)
            self.log_step = self.log_file['#"Step"'].iloc[-1]

            # Checks to see if we have enough steps in the log file to start validation
            if len(self.log_file) < c.MIN_LOGGING_ENTRIES:
                raise ValidationError(
                    f"Miner {self.hotkey_alias} did not run enough steps in the simulation... Skipping!"
                )

            # Make sure that we are enough steps ahead in the log file compared to the checkpoint file.
            # Checks if log_file is MIN_STEPS steps ahead of checkpoint
            if (self.log_step - self.simulation.currentStep) < c.MIN_SIMULATION_STEPS:
                # If the miner did not run enough steps, we will load the old checkpoint
                checkpoint_path = os.path.join(self.miner_data_directory, f"{self.current_state}_old.cpt")
                if os.path.exists(checkpoint_path):
                    logger.warning(
                        f"Miner {self.hotkey_alias} did not run enough steps since last checkpoint... Loading old checkpoint"
                    )
                    self.simulation.loadCheckpoint(checkpoint_path)
                    # Checking to see if the old checkpoint has enough steps to validate
                    if (self.log_step - self.simulation.currentStep) < c.MIN_SIMULATION_STEPS:
                        raise ValidationError(
                            f"Miner {self.hotkey_alias} did not run enough steps in the simulation... Skipping!"
                        )
                else:
                    raise ValidationError(
                        f"Miner {self.hotkey_alias} did not run enough steps and no old checkpoint found... Skipping!"
                    )

            self.cpt_step = self.simulation.currentStep
            self.checkpoint_path = checkpoint_path
            self.state_xml_path = state_xml_path

            # Create the state file here because it could have been loaded after MIN_SIMULATION_STEPS check
            self.simulation.saveState(self.state_xml_path)

            # Save the system config to the miner data directory
            system_config_path = os.path.join(self.miner_data_directory, f"miner_system_config_{seed}.pkl")
            if not os.path.exists(system_config_path):
                write_pkl(
                    data=self.system_config,
                    path=system_config_path,
                    write_mode="wb",
                )

        except ValidationError as E:
            logger.warning(f"{E}")
            return False

        except Exception as e:
            logger.error(f"Failed to recreate simulation: {e}")
            return False

        return True

    def check_masses(self, velm_array_location: str) -> bool:
        """
        Check if the masses reported in the miner file are identical to the masses given
        in the initial pdb file. If not, they have modified the system in unintended ways.

        Reference:
        https://github.com/openmm/openmm/blob/53770948682c40bd460b39830d4e0f0fd3a4b868/platforms/common/src/kernels/langevinMiddle.cc#L11
        """

        validator_velm_data = load_pkl(velm_array_location, "rb")
        miner_velm_data = create_velm(simulation=self.simulation)

        validator_masses = validator_velm_data["pdb_masses"]
        miner_masses = miner_velm_data["pdb_masses"]

        for i, (v_mass, m_mass) in enumerate(zip(validator_masses, miner_masses)):
            if v_mass != m_mass:
                logger.error(f"Masses for atom {i} do not match. Validator: {v_mass}, Miner: {m_mass}")
                return False
        return True

    def check_gradient(self, check_energies: np.ndarray) -> True:
        """This method checks the gradient of the potential energy within the first
        WINDOW size of the check_energies array. Miners that return gradients that are too high,
        there is a *high* probability that they have not run the simulation as the validator specified.
        """
        WINDOW = 50  # Number of steps to calculate the gradient over
        GRADIENT_THRESHOLD = 10  # kJ/mol/nm

        mean_gradient = np.diff(check_energies[:WINDOW]).mean().item()
        return mean_gradient <= GRADIENT_THRESHOLD  # includes large negative gradients is passible

    def compare_state_to_cpt(self, state_energies: list, checkpoint_energies: list) -> bool:
        """
        Check if the state file is the same as the checkpoint file by comparing the median of the first few energy values
        in the simulation created by the checkpoint and the state file respectively.
        """

        WINDOW = 50

        state_energies = np.array(state_energies)
        checkpoint_energies = np.array(checkpoint_energies)

        state_median = np.median(state_energies[:WINDOW])
        checkpoint_median = np.median(checkpoint_energies[:WINDOW])

        percent_diff = abs((state_median - checkpoint_median) / checkpoint_median) * 100

        if percent_diff > c.XML_CHECKPOINT_THRESHOLD:
            return False
        return True

    def is_run_valid(self):
        """
        Checks if the run is valid by evaluating a set of logical conditions:

        1. comparing the potential energy values between the current simulation and a reference log file.
        2. ensuring that the gradient of the minimization is within a certain threshold to prevent exploits.
        3. ensuring that the masses of the atoms in the simulation are the same as the masses in the original pdb file.


        Returns:
            Tuple[bool, list, list]: True if the run is valid, False otherwise.
                The two lists contain the potential energy values from the current simulation and the reference log file.
        """

        steps_to_run = min(c.MAX_SIMULATION_STEPS_FOR_EVALUATION, self.log_step - self.cpt_step)

        # This is where we are going to check the xml files for the state.
        logger.info(f"Recreating simulation for {self.pdb_id} for state-based analysis...")
        self.simulation, self.system_config = self.md_simulator.create_simulation(
            pdb=load_pdb_file(pdb_file=self.pdb_location),
            system_config=self.system_config.get_config(),
            seed=self.miner_seed,
        )
        self.simulation.loadState(self.state_xml_path)
        state_energies = []
        for _ in range(steps_to_run // 10):
            self.simulation.step(10)
            energy = self.simulation.context.getState(getEnergy=True).getPotentialEnergy()._value
            state_energies.append(energy)

        try:
            if not self.check_gradient(check_energies=state_energies):
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed state-gradient check for {self.pdb_id}, ... Skipping!"
                )
                raise "state-gradient"

            # Reload in the checkpoint file and run the simulation for the same number of steps as the miner.
            self.simulation, self.system_config = self.md_simulator.create_simulation(
                pdb=load_pdb_file(pdb_file=self.pdb_location),
                system_config=self.system_config.get_config(),
                seed=self.miner_seed,
            )
            self.simulation.loadCheckpoint(self.checkpoint_path)

            current_state_logfile = os.path.join(self.miner_data_directory, f"check_{self.current_state}.log")
            self.simulation.reporters.append(
                app.StateDataReporter(
                    current_state_logfile,
                    10,
                    step=True,
                    potentialEnergy=True,
                )
            )

            logger.info(f"Running {steps_to_run} steps. log_step: {self.log_step}, cpt_step: {self.cpt_step}")

            max_step = self.cpt_step + steps_to_run
            miner_energies: np.ndarray = self.log_file[
                (self.log_file['#"Step"'] > self.cpt_step) & (self.log_file['#"Step"'] <= max_step)
            ]["Potential Energy (kJ/mole)"].values

            self.simulation.step(steps_to_run)

            check_log_file = pd.read_csv(current_state_logfile)
            check_energies: np.ndarray = check_log_file["Potential Energy (kJ/mole)"].values

            if len(np.unique(check_energies)) == 1:
                logger.warning("All energy values in reproduced simulation are the same. Skipping!")
                raise "reprod-energies-identical"

            if not self.check_gradient(check_energies=check_energies):
                logger.warning(f"hotkey {self.hotkey_alias} failed cpt-gradient check for {self.pdb_id}, ... Skipping!")
                raise "cpt-gradient"

            if not self.compare_state_to_cpt(state_energies=state_energies, checkpoint_energies=check_energies):
                logger.warning(
                    f"hotkey {self.hotkey_alias} failed state-checkpoint comparison for {self.pdb_id}, ... Skipping!"
                )
                raise "state-checkpoint"

            # calculating absolute percent difference per step
            percent_diff = abs(((check_energies - miner_energies) / miner_energies) * 100)
            median_percent_diff = np.median(percent_diff)

            if median_percent_diff > c.ANOMALY_THRESHOLD:
                raise "anomaly"

            # Save the folded pdb file if the run is valid
            positions = self.simulation.context.getState(getPositions=True).getPositions()
            topology = self.simulation.topology

            save_pdb(
                positions=positions,
                topology=topology,
                output_path=os.path.join(self.miner_data_directory, f"{self.pdb_id}_folded.pdb"),
            )

            return True, check_energies.tolist(), miner_energies.tolist(), "valid"

        except Exception as E:
            return False, [], [], str(E)

    def _evaluate(self, data: Dict[str, Any]) -> float:
        if not self.process_md_output(**data):
            return 0.0

        if not self.check_masses(velm_array_location=data["velm_array_location"]):
            return 0.0

        # Check to see if we have a logging resolution of 10 or better, if not the run is not valid
        if (self.log_file['#"Step"'][1] - self.log_file['#"Step"'][0]) > 10:
            return 0.0

        is_valid, checked_energies, miner_energies, result = self.is_run_valid()
        if not is_valid:
            return 0.0

        return np.median(checked_energies[-10:])  # Last portion of the reproduced energy vector

    def name(self) -> str:
        return "SyntheticMDReward"


class OrganicMDEvaluator(BaseEvaluator):
    def __init__(self, priority: float = 1, **kwargs):
        self.priority = priority
        self.kwargs = kwargs

    def _evaluate(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "OrganicMDReward"


class SyntheticMLEvaluator(BaseEvaluator):
    def __init__(self, priority: float = 1, **kwargs):
        self.priority = priority
        self.kwargs = kwargs

    def _evaluate(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "SyntheticMLReward"


class OrganicMLEvaluator(BaseEvaluator):
    def __init__(self, priority: float = 1, **kwargs):
        self.priority = priority
        self.kwargs = kwargs

    def _evaluate(self, data: Dict[str, Any]) -> float:
        return 0.0

    def name(self) -> str:
        return "OrganicMLReward"


class EvaluationRegistry:
    """
    Handles the organization of all tasks that we want inside of SN25, which includes:
        - Molecular Dynamics (MD)
        - ML Inference

    It also attaches its corresponding reward pipelines.
    """

    def __init__(self):
        evaluation_pipelines = [SyntheticMDEvaluator, OrganicMDEvaluator, SyntheticMLEvaluator, OrganicMLEvaluator]

        self.tasks = []
        for pipe in evaluation_pipelines:
            self.tasks.append(pipe().name())
