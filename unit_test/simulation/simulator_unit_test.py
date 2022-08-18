import os
import unittest
import numpy as np

from QuICT.simulation.unitary import UnitarySimulator
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.simulation.density_matrix import DensityMatrixSimulation
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
from QuICT.simulation import Simulator

@unittest.skipUnless(os.environ.get("test_with_gpu", True), "require GPU")
class TestGPUSimulator(unittest.TestCase): 
    @classmethod
    def setUpClass(cls): 
        print('GPU simulator unit test begin!')
        # Import the data required for testing
        cls.qasm = OPENQASMInterface.load_file(os.path.dirname(os.path.abspath(__file__)) + "/data/random_circuit_for_correction.qasm")
        cls.circuit = cls.qasm.circuit
        cls.sv_data = np.load(os.path.dirname(os.path.abspath(__file__)) + "/data/state_vector.npy")
        cls.dm_data = np.load(os.path.dirname(os.path.abspath(__file__)) + "/data/density_matrix.npy")

    @classmethod
    def tearDownClass(cls): 
        print('GPU simulator unit test finished!')
  
    def test_unitary(self):
        sim = UnitarySimulator("GPU")
        U = sim.run(TestGPUSimulator.circuit).get()
        assert np.allclose(U, TestGPUSimulator.sv_data)

        u_sim = Simulator(device="GPU")
        u = u_sim.run(TestGPUSimulator.circuit)
        assert np.allclose(u["data"]["state_vector"], TestGPUSimulator.sv_data)

    def test_state_vector(self):
        sim = ConstantStateVectorSimulator("double")
        SV = sim.run(TestGPUSimulator.circuit).get()
        assert np.allclose(SV, TestGPUSimulator.sv_data)

        sv_sim = Simulator(device="GPU")
        sv = sv_sim.run(TestGPUSimulator.circuit)
        assert np.allclose(sv["data"]["state_vector"], TestGPUSimulator.sv_data)

    
    def test_densitymatrix(self):
        sim = DensityMatrixSimulation("GPU")
        DM = sim.run(TestGPUSimulator.circuit).get()
        assert np.allclose(DM,TestGPUSimulator.dm_data)

        d_sim = Simulator(device="GPU", backend="density_matrix")
        dm = d_sim.run(TestGPUSimulator.circuit)
        print(type(dm["data"]["density_matrix"]))
        assert np.allclose(dm["data"]["density_matrix"], TestGPUSimulator.dm_data)


class TestCPUSimulator(unittest.TestCase): 
    @classmethod
    def setUpClass(cls): 
        print('CPU simulator unit test begin!')
        # Import the data required for testing
        cls.qasm = OPENQASMInterface.load_file(os.path.dirname(os.path.abspath(__file__)) + "/data/random_circuit_for_correction.qasm")
        cls.circuit = cls.qasm.circuit
        cls.sv_data = np.load(os.path.dirname(os.path.abspath(__file__)) + "/data/state_vector.npy")
        cls.dm_data = np.load(os.path.dirname(os.path.abspath(__file__)) + "/data/density_matrix.npy")

    @classmethod
    def tearDownClass(cls): 
        print('CPU simulator unit test finished!')
    
    def test_unitary(self):
        sim = UnitarySimulator()
        U = sim.run(TestCPUSimulator.circuit)
        assert np.allclose(U, TestCPUSimulator.sv_data)

        u_sim = Simulator(device="CPU")
        u = u_sim.run(TestCPUSimulator.circuit)
        assert np.allclose(u["data"]["state_vector"], TestCPUSimulator.sv_data)

    def test_state_vector(self):
        sim = ConstantStateVectorSimulator("double")
        SV = sim.run(TestCPUSimulator.circuit)
        assert np.allclose(SV, TestCPUSimulator.sv_data)

        sv_sim = Simulator(device="CPU")
        sv = sv_sim.run(TestCPUSimulator.circuit)
        assert np.allclose(sv["data"]["state_vector"], TestCPUSimulator.sv_data)

    def test_densitymatrix(self):
        simulator = DensityMatrixSimulation()
        DM = simulator.run(TestCPUSimulator.circuit)
        assert np.allclose(DM,TestCPUSimulator.dm_data)

        d_sim = Simulator(device="CPU",backend="density_matrix")
        dm = d_sim.run(TestCPUSimulator.circuit)
        assert np.allclose(dm["data"]["density_matrix"], TestCPUSimulator.dm_data)
        
if __name__=="__main__":
  unittest.main()


    