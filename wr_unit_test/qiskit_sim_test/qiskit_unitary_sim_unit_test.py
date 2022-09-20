
import numpy as np
from QuICT.algorithm.synthetical_unitary.synthetical_unitary import SyntheticalUnitary
from QuICT.core.gate.gate import Unitary
from QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator
from QuICT.simulation.unitary.unitary_simulator import UnitarySimulator
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
from QuICT.simulation import simulator




def qiskitunitary():
    qasm = OPENQASMInterface.load_file("QuICT/wr_unit_test/qiskit_sim_test/random_circuit_for_correction.qasm")
    circuit = qasm.circuit

    # circuit_unitary = SyntheticalUnitary.run(circuit)
    # result_mat = simulator.get_unitary_matrix(circuit)
    # assert np.allclose(circuit_unitary,result_mat)
    
    sim = UnitarySimulator("GPU")
    U = sim.run(circuit).get()
    new_data = np.load("QuICT/wr_unit_test/qiskit_sim_test/temp.npy")
    print(np.allclose(U, new_data))

   

qiskitunitary()