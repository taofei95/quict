
import numpy as np
from QuICT.core.gate.gate import Measure
from QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
  
def qiskitstatevector():
    
    qasm = OPENQASMInterface.load_file("QuICT/wr_unit_test/qiskit_sim_test/random_circuit_for_correction.qasm")
    circuit = qasm.circuit


    sim = ConstantStateVectorSimulator()
    sv = sim.run(circuit).get()
    new_data = np.load("QuICT/wr_unit_test/qiskit_sim_test/temp.npy")
    print(np.allclose(sv, new_data))

qiskitstatevector()
