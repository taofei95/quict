import unittest
import numpy as np
from QuICT.simulation.density_matrix.density_matrix_simulator import DensityMatrixSimulation
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator
from QuICT.simulation.unitary.unitary_simulator import UnitarySimulator
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

class TestQiskit(unittest.TestCase): 
    @classmethod
    def setUpClass(cls): 
        print('Qiskit simulator begin')
        qasm = OPENQASMInterface.load_file("QuICT/wr_unit_test/qiskit_sim_test/random_circuit_for_correction.qasm")
        circuit = qasm.circuit
        new_data = np.load("QuICT/wr_unit_test/qiskit_sim_test/temp.npy")
    @classmethod
    def tearDownClass(cls): 
        print('Qiskit simulator finished')
    
    #device="GPU"
    def qiskitunitary(self):
        sim = UnitarySimulator("GPU")
        U = sim.run(TestQiskit.circuit).get()
        assert np.allclose(U, TestQiskit.new_data)

    def qiskitstatevector(self):
        sim = ConstantStateVectorSimulator("GPU")
        sv = sim.run(TestQiskit.circuit).get()
        assert np.allclose(sv, TestQiskit.new_data)
    
    def qiskit_dm_simulator(self):
        simulator = DensityMatrixSimulation("GPU")
        dm = simulator.run(TestQiskit.circuit).get()
        assert np.allclose(dm,TestQiskit.new_data)

    #device="CPU"
    def qiskitunitary(self):
        sim = UnitarySimulator()
        U = sim.run(TestQiskit.circuit).get()
        assert np.allclose(U, TestQiskit.new_data)

    def qiskitstatevector(self):
        sim = ConstantStateVectorSimulator()
        sv = sim.run(TestQiskit.circuit).get()
        assert np.allclose(sv, TestQiskit.new_data)
    
    def qiskit_dm_simulator(self):
        simulator = DensityMatrixSimulation()
        dm = simulator.run(TestQiskit.circuit).get()
        assert np.allclose(dm,TestQiskit.new_data)

if __name__=="__main__":
  unittest.main()


    