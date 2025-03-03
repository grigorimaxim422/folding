from openmm.app import *
from openmm import *
from openmm.unit import *
import openmm as mm
import logging
import time
from sys import stdout
import os

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


def run_simulation(gpu_id):
    """Runs an OpenMM simulation on the specified GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Assign specific GPU
    
    pdb = PDBFile("1AKI.pdb")
    forcefield = ForceField('amber14-all.xml', 'amber14/tip3pfb.xml')
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.deleteWater()
    residues=modeller.addHydrogens(forcefield)

    modeller.addSolvent(forcefield, padding=1.0*nanometer)

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=PME, nonbondedCutoff=1.0*nanometer, constraints=HBonds)
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.004*picoseconds)

    platform = mm.Platform.getPlatformByName("CUDA")

    # # Reference for DisablePmeStream: https://github.com/openmm/openmm/issues/3589
    # deviceIndex = first_number(get_idle_gpus())        
    # if deviceIndex == -1:
    #     deviceIndex = static_id
    #     static_id = static_id + 1
    #     static_id = static_id % 4
        
    # logging.info(f"Picked {deviceIndex}...")
    properties = {
        # "DeterministicForces": "true",
        "Precision": "single",
        # "DisablePmeStream": "true",
        "CudaPrecision":"single",
        "DeviceIndex":"0",
        "CudaDeviceIndex":"0"
    }
    # properties['DeviceIndex'] = f"{deviceIndex}"
    # properties['CudaDeviceIndex'] = f"{deviceIndex}"


    simulation = Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)


    print("Minimizing energy")
    simulation.minimizeEnergy()


    simulation.reporters.append(PDBReporter('output.pdb', 1000))
    simulation.reporters.append(StateDataReporter(stdout, 1000, step=True,
            potentialEnergy=True, temperature=True, volume=True))
    simulation.reporters.append(StateDataReporter("md_log.txt", 100, step=True,
            potentialEnergy=True, temperature=True, volume=True))

    print("Running NVT")
    simulation.step(100000)

    system.addForce(MonteCarloBarostat(1*bar, 300*kelvin))
    simulation.context.reinitialize(preserveState=True)


    print("Running NPT")
    simulation.step(100000)
    

import time
import multiprocessing

if __name__ == "__main__":
    # Start first simulation on GPU 0
    for i in [0,1]:
        process1 = multiprocessing.Process(target=run_simulation, args=(i,))
        process1.start()
