import os
import unittest
from QuICT.core import Circuit
from QuICT.simulation import Simulator
from QuICT.simulation.density_matrix import density_matrix_simulator
from QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator
from QuICT.simulation.unitary import unitary_simulator
from QuICT.core import circuit

class TestSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls): 
        print('GPU CPU Simlator begin!')
        cls.circuit = Circuit(10)
        cls.circuit.random_append(100)
    
    @classmethod
    def tearDownClass(cls): 
        print('GPU CPU Simlator finished!')

    #device="GPU"
    def test_gpu_unitary(self):
        u_sim = Simulator(
            device="GPU",
            backend="unitary"
        )
        _ = u_sim.run(TestSimulator.circuit)

        assert 1
    
    def test_gpu_statevector(self):
        sv_sim = Simulator(
            device="GPU",
            backend="statevector"
        )

        _ = sv_sim.run(TestSimulator.circuit)

        assert 1

    def test_gpu_density_matrix(self):
        d_sim = Simulator(
            device="GPU",
            backend="density_matrix"
        )
        _ = d_sim.run(TestSimulator.circuit)

        assert 1


    # device="CPU"
    def test_cpu_unitary(self):
        u_sim = Simulator(
            device="CPU",
            backend="unitary"
        )
        _ = u_sim.run(TestSimulator.circuit)

        assert 1

    def test_cpu_statevector(self):
        sv_sim = Simulator(
            device="CPU",
            backend="statevector"
        )

        _ = sv_sim.run(TestSimulator.circuit)
        assert 1

    def test_cpu_density_matrix(self):
        d_sim = Simulator(
            device="CPU",
            backend="density_matrix"
        )
        _ = d_sim.run(TestSimulator.circuit)

        assert 1


if __name__ ==" __main__":
    unittest.main()


