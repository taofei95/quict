import numpy as np
from QuICT.core.gate.gate import Measure
from QuICT.simulation.density_matrix.density_matrix_simulator import DensityMatrixSimulation
from QuICT.tools.interface.qasm_interface import OPENQASMInterface



 
def qiskit_dm_simulator():
    qasm = OPENQASMInterface.load_file("QuICT/wr_unit_test/qiskit_sim_test/random_circuit_for_correction.qasm")
    circuit = qasm.circuit


    simulator = DensityMatrixSimulation("GPU")
    dm = simulator.run(circuit).get()
    new_data = np.load("QuICT/wr_unit_test/qiskit_sim_test/density_matrix.npy")
   
    print(np.allclose(dm,new_data))


qiskit_dm_simulator()
