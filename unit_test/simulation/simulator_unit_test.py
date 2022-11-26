import os
import unittest
import numpy as np
from copy import deepcopy

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.unitary import UnitarySimulator
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.simulation.density_matrix import DensityMatrixSimulation
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
from QuICT.simulation import Simulator


@unittest.skipUnless(os.environ.get("test_with_gpu", False), "require GPU")
class TestGPUSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('GPU simulator unit test begin!')
        # Import the data required for testing
        cls.qasm = OPENQASMInterface.load_file(
            os.path.dirname(os.path.abspath(__file__)) + "/data/random_circuit_for_correction.qasm"
        )
        cls.circuit = cls.qasm.circuit
        cls.sv_data = np.load(os.path.dirname(os.path.abspath(__file__)) + "/data/state_vector.npy")
        cls.sv_data_single = cls.sv_data.astype(np.complex64)
        cls.dm_data = np.load(os.path.dirname(os.path.abspath(__file__)) + "/data/density_matrix.npy")
        cls.dm_data_single = cls.dm_data.astype(np.complex64)

    @classmethod
    def tearDownClass(cls):
        print('GPU simulator unit test finished!')

    def test_unitary(self):
        sim = UnitarySimulator("GPU")
        U = sim.run(deepcopy(TestGPUSimulator.circuit))
        assert np.allclose(U, TestGPUSimulator.sv_data)

        sim = UnitarySimulator("GPU", precision="single")
        U = sim.run(deepcopy(TestGPUSimulator.circuit))
        assert np.allclose(U, TestGPUSimulator.sv_data_single, atol=1e-6)

        u_sim = Simulator(device="GPU", backend="unitary")
        u = u_sim.run(deepcopy(TestGPUSimulator.circuit))
        assert np.allclose(u["data"]["state_vector"], TestGPUSimulator.sv_data)

        u_sim = Simulator(device="GPU", precision="single", backend="unitary")
        u = u_sim.run(deepcopy(TestGPUSimulator.circuit))
        assert np.allclose(u["data"]["state_vector"], TestGPUSimulator.sv_data_single, atol=1e-6)

    def test_state_vector(self):
        sim = ConstantStateVectorSimulator("double")
        SV = sim.run(deepcopy(TestGPUSimulator.circuit)).get()
        assert np.allclose(SV, TestGPUSimulator.sv_data)

        sim = ConstantStateVectorSimulator("single")
        SV = sim.run(deepcopy(TestGPUSimulator.circuit)).get()
        assert np.allclose(SV, TestGPUSimulator.sv_data_single, atol=1e-6)

        sv_sim = Simulator(device="GPU")
        sv = sv_sim.run(deepcopy(TestGPUSimulator.circuit))
        assert np.allclose(sv["data"]["state_vector"], TestGPUSimulator.sv_data)

        sv_sim = Simulator(device="GPU", precision="single")
        sv = sv_sim.run(deepcopy(TestGPUSimulator.circuit))
        assert np.allclose(sv["data"]["state_vector"], TestGPUSimulator.sv_data_single, atol=1e-6)

    def test_density_matrix(self):
        sim = DensityMatrixSimulation("GPU")
        DM = sim.run(deepcopy(TestGPUSimulator.circuit)).get()
        assert np.allclose(DM, TestGPUSimulator.dm_data)

        sim = DensityMatrixSimulation("GPU", precision="single")
        DM = sim.run(deepcopy(TestGPUSimulator.circuit)).get()
        assert np.allclose(DM, TestGPUSimulator.dm_data_single, atol=1e-6)

        d_sim = Simulator(device="GPU", backend="density_matrix")
        dm = d_sim.run(deepcopy(TestGPUSimulator.circuit))
        assert np.allclose(dm["data"]["density_matrix"], TestGPUSimulator.dm_data)

        d_sim = Simulator(device="GPU", backend="density_matrix", precision="single")
        dm = d_sim.run(deepcopy(TestGPUSimulator.circuit))
        assert np.allclose(dm["data"]["density_matrix"], TestGPUSimulator.dm_data_single, atol=1e-6)

    def test_matrix_aggregation(self):
        t = ConstantStateVectorSimulator(matrix_aggregation=True)
        T = t.run(deepcopy(TestGPUSimulator.circuit)).get()
        assert np.allclose(T, TestGPUSimulator.sv_data)

        t = ConstantStateVectorSimulator(matrix_aggregation=True, precision="single")
        T = t.run(deepcopy(TestGPUSimulator.circuit)).get()
        assert np.allclose(T, TestGPUSimulator.sv_data_single, atol=1e-6)

        f = ConstantStateVectorSimulator(matrix_aggregation=False)
        F = f.run(deepcopy(TestGPUSimulator.circuit)).get()
        assert np.allclose(F, TestGPUSimulator.sv_data)

        f = ConstantStateVectorSimulator(matrix_aggregation=False, precision="single")
        F = f.run(deepcopy(TestGPUSimulator.circuit)).get()
        assert np.allclose(F, TestGPUSimulator.sv_data_single, atol=1e-6)


class TestCPUSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('CPU simulator unit test begin!')
        # Import the data required for testing
        cls.qasm = OPENQASMInterface.load_file(
            os.path.dirname(os.path.abspath(__file__)) + "/data/random_circuit_for_correction.qasm"
        )
        cls.circuit = cls.qasm.circuit
        cls.sv_data = np.load(os.path.dirname(os.path.abspath(__file__)) + "/data/state_vector.npy")
        cls.dm_data = np.load(os.path.dirname(os.path.abspath(__file__)) + "/data/density_matrix.npy")

    @classmethod
    def tearDownClass(cls):
        print('CPU simulator unit test finished!')

    def test_unitary(self):
        sim = UnitarySimulator()
        U = sim.run(deepcopy(TestCPUSimulator.circuit))
        assert np.allclose(U, TestCPUSimulator.sv_data)

        u_sim = Simulator(device="CPU", backend="unitary")
        u = u_sim.run(deepcopy(TestCPUSimulator.circuit))
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
        DM = simulator.run(deepcopy(TestCPUSimulator.circuit))
        assert np.allclose(DM, TestCPUSimulator.dm_data)

        d_sim = Simulator(device="CPU", backend="density_matrix")
        dm = d_sim.run(deepcopy(TestCPUSimulator.circuit))
        assert np.allclose(dm["data"]["density_matrix"], TestCPUSimulator.dm_data)


class TestSample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('simulator sample test begin!')
        cls.qubit_num = 4
        cls.cir = Circuit(cls.qubit_num)

    @classmethod
    def tearDownClass(cls):
        print('simulator sample test finished!')

    def test_sample(self):
        H | TestSample.cir(0)
        for i in range(TestSample.qubit_num - 1):
            CX | TestSample.cir([i, i + 1])

        # double
        simulator = DensityMatrixSimulation()
        _ = simulator.run(TestSample.cir)
        a = simulator.sample(100)
        assert a[0] + a[-1] == 100

        simulator = UnitarySimulator()
        _ = simulator.run(TestSample.cir)
        b = simulator.sample(100)
        assert b[0] + b[-1] == 100

        simulator = ConstantStateVectorSimulator()
        _ = simulator.run(TestSample.cir)
        c = simulator.sample(100)
        assert c[0] + c[-1] == 100

        # single
        simulator = DensityMatrixSimulation(precision="single")
        _ = simulator.run(TestSample.cir)
        d = simulator.sample(100)
        assert d[0] + d[-1] == 100

        simulator = UnitarySimulator(precision="single")
        _ = simulator.run(TestSample.cir)
        b = simulator.sample(100)
        assert b[0] + b[-1] == 100

        simulator = ConstantStateVectorSimulator(precision="single")
        _ = simulator.run(TestSample.cir)
        c = simulator.sample(100)
        assert c[0] + c[-1] == 100

        # make the running times as shots
        sim = Simulator(device="GPU", shots=100)
        s = sim.run(TestSample.cir)
        a = s["data"]["counts"]
        assert a['0000'] + a['1111'] == 100


if __name__ == "__main__":
    unittest.main()
