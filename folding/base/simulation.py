from abc import ABC, abstractmethod
import time
from typing import Tuple
import functools
import openmm as mm
from openmm import app
from openmm import unit
import torch

from folding.utils.opemm_simulation_config import SimulationConfig
from folding.utils.logger import logger

import torch

def get_idle_gpus(threshold=0.05):
    """
    Checks GPU memory usage to estimate idle GPUs.
    
    Args:
        threshold (float): Fraction of total memory usage below which GPU is considered idle.
    
    Returns:
        List[int]: List of idle GPU indices.
    """
    idle_gpus = []
    num_gpus = torch.cuda.device_count()
    
    for i in range(num_gpus):
        torch.cuda.set_device(i)
        total_memory = torch.cuda.get_device_properties(i).total_memory
        used_memory = torch.cuda.memory_allocated(i)
        
        if (used_memory / total_memory) < threshold:
            idle_gpus.append(i)
    
    return idle_gpus

#print("Idle GPUs:", get_idle_gpus())

def first_number(arr):
    return arr[0] if arr else -1

class GenericSimulation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def create_simulation(self):
        pass

    @staticmethod
    def timeit(method):
        @functools.wraps(method)
        def timed(*args, **kwargs):
            start_time = time.time()
            result = method(*args, **kwargs)
            end_time = time.time()
            logger.info(
                f"Method {method.__name__} took {end_time - start_time:.4f} seconds"
            )
            return result

        return timed


class OpenMMSimulation(GenericSimulation):
    static_id = 0
    
    @GenericSimulation.timeit
    def create_simulation(
        self, pdb: app.PDBFile, system_config: dict, seed: int = None, verbose=False
    ) -> Tuple[app.Simulation, SimulationConfig]:
        """Recreates a simulation object based on the provided parameters.

        This method takes in a seed, state, and checkpoint file path to recreate a simulation object.
        Args:
            seed (str): The seed for the random number generator.
            system_config (dict): A dictionary containing the system configuration settings.
            pdb (app.PDBFile): The PDB file used to initialize the simulation

        Returns:
        Tuple[app.Simulation, SimulationConfig]: A tuple containing the recreated simulation object and the potentially altered system configuration in SystemConfig format.
        """
        setup_times = {}

        start_time = time.time()
        forcefield = app.ForceField(system_config["ff"], system_config["water"])
        setup_times["add_ff"] = time.time() - start_time

        modeller = app.Modeller(pdb.topology, pdb.positions)

        start_time = time.time()
        modeller.deleteWater()
        setup_times["delete_water"] = time.time() - start_time

        # modeller.addExtraParticles(forcefield)

        start_time = time.time()
        modeller.addHydrogens(forcefield)
        setup_times["add_hydrogens"] = time.time() - start_time

        start_time = time.time()
        modeller.addSolvent(
            forcefield,
            padding=system_config["box_padding"] * unit.nanometer,
            boxShape=system_config["box"],
        )
        setup_times["add_solvent"] = time.time() - start_time

        # Create the system
        start_time = time.time()
        # The assumption here is that the system_config cutoff MUST be given in nanometers
        threshold = (
            pdb.topology.getUnitCellDimensions().min().value_in_unit(mm.unit.nanometers)
        ) / 2
        if system_config["cutoff"] > threshold:
            nonbondedCutoff = threshold * mm.unit.nanometers
            # set the attribute in the config for the pipeline.
            system_config["cutoff"] = threshold
            logger.debug(
                f"Nonbonded cutoff is greater than half the minimum box dimension. Setting nonbonded cutoff to {threshold} nm"
            )
        else:
            nonbondedCutoff = system_config["cutoff"] * mm.unit.nanometers

        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=mm.app.NoCutoff,
            nonbondedCutoff=nonbondedCutoff,
            constraints=system_config["constraints"],
        )
        setup_times["create_system"] = time.time() - start_time

        # Integrator settings
        integrator = mm.LangevinIntegrator(
            system_config["temperature"] * unit.kelvin,
            system_config["friction"] / unit.picosecond,
            system_config["time_step_size"] * unit.picoseconds,
        )

        seed = seed if seed is not None else system_config["seed"]
        integrator.setRandomNumberSeed(seed)

        # Periodic boundary conditions
        # pdb.topology.setPeriodicBoxVectors(system.getDefaultPeriodicBoxVectors())

        # if state != "nvt":
        #     system.addForce(
        #         mm.MonteCarloBarostat(
        #             system_config["pressure"] * unit.bar,
        #             system_config["temperature"] * unit.kelvin,
        #         )
        #     )

        platform = mm.Platform.getPlatformByName("CUDA")

        # Reference for DisablePmeStream: https://github.com/openmm/openmm/issues/3589
        deviceIndex = first_number(get_idle_gpus())        
        if deviceIndex == -1:
            deviceIndex = static_id
            static_id = static_id + 1
            static_id = static_id % 4
            
        logging.info(f"Picked {deviceIndex}...")
        properties = {
            "DeterministicForces": "true",
            "Precision": "double",
            "DisablePmeStream": "true",
            "CudaPrecision":"mixed",
            # "DeviceIndex":"0,1"
        }
        properties['DeviceIndex'] = f"{deviceIndex}"
        properties['CudaDeviceIndex'] = f"{deviceIndex}"

        start_time = time.time()
        simulation = mm.app.Simulation(
            modeller.topology, system, integrator, platform, properties
        )
        setup_times["create_simulation"] = time.time() - start_time
        # Set initial positions

        start_time = time.time()
        simulation.context.setPositions(modeller.positions)
        setup_times["set_positions"] = time.time() - start_time

        # Converting the system config into a Dict[str,str] and ensure all values in system_config are of the correct type
        for k, v in system_config.items():
            if not isinstance(v, (str, int, float, dict)):
                system_config[k] = str(v)

        if verbose:
            for key, t in setup_times:
                logger.debug(f"Took {round(t, 3)} to {key}")

        return simulation, SimulationConfig(**system_config)
