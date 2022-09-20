import unittest
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.density_matrix import DensityMatrixSimulation
from QuICT.core import qubit
from QuICT.simulation.state_vector.cpu_simulator.cpu import CircuitSimulator


class TestDensityMatrixSimulator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Density Matrix Simulator unit test start!")
        qubit_num=3
        cls.circuit = Circuit(qubit_num)
        H | cls.circuit
        CX | cls.circuit([1, 0])
        CX | cls.circuit([0, 1])
        CH | cls.circuit([1, 0])
        Swap | cls.circuit([2, 1])
        SX | cls.circuit(0)
        T | cls.circuit(1)
        T_dagger | cls.circuit(0)
        X | cls.circuit(1)
        Y | cls.circuit(1)
        S | cls.circuit(2)
        U1(np.pi / 2) | cls.circuit(2)
        U3(np.pi, 0, 1) | cls.circuit(0)
        Rx(np.pi) | cls.circuit(1)
        Ry(np.pi / 2) | cls.circuit(2)
        Rz(np.pi / 4) | cls.circuit(0)

        cls.result = np.array(
            [[2.13388348e-01 + 0.j, 2.42861287e-17 - 0.08838835j,
              -1.50888348e-01 - 0.15088835j, -6.25000000e-02 + 0.0625j,
              -4.54428443e-02 + 0.20849349j, -8.63608307e-02 - 0.01882304j,
              1.79560103e-01 - 0.11529422j, 4.77564280e-02 + 0.07437623j],
             [2.42861287e-17 + 0.08838835j, 3.66116524e-02 + 0.j,
              6.25000000e-02 - 0.0625j, -2.58883476e-02 - 0.02588835j,
              -8.63608307e-02 - 0.01882304j, 7.79675946e-03 - 0.03577183j,
              4.77564280e-02 + 0.07437623j, -3.08076432e-02 + 0.01978136j],
             [-1.50888348e-01 + 0.15088835j, 6.25000000e-02 + 0.0625j,
              2.13388348e-01 + 0.j, 2.77555756e-17 - 0.08838835j,
              -1.15294216e-01 - 0.1795601j, 7.43762299e-02 - 0.04775643j,
              -4.54428443e-02 + 0.20849349j, -8.63608307e-02 - 0.01882304j],
             [-6.25000000e-02 - 0.0625j, -2.58883476e-02 + 0.02588835j,
              2.77555756e-17 + 0.08838835j, 3.66116524e-02 + 0.j,
              7.43762299e-02 - 0.04775643j, 1.97813602e-02 + 0.03080764j,
              -8.63608307e-02 - 0.01882304j, 7.79675946e-03 - 0.03577183j],
             [-4.54428443e-02 - 0.20849349j, -8.63608307e-02 + 0.01882304j,
              -1.15294216e-01 + 0.1795601j, 7.43762299e-02 + 0.04775643j,
              2.13388348e-01 + 0.j, 2.45105275e-17 + 0.08838835j,
              -1.50888348e-01 - 0.15088835j, 6.25000000e-02 - 0.0625j],
             [-8.63608307e-02 + 0.01882304j, 7.79675946e-03 + 0.03577183j,
              7.43762299e-02 + 0.04775643j, 1.97813602e-02 - 0.03080764j,
              2.45105275e-17 - 0.08838835j, 3.66116524e-02 + 0.j,
              -6.25000000e-02 + 0.0625j, -2.58883476e-02 - 0.02588835j],
             [1.79560103e-01 + 0.11529422j, 4.77564280e-02 - 0.07437623j,
              -4.54428443e-02 - 0.20849349j, -8.63608307e-02 + 0.01882304j,
              -1.50888348e-01 + 0.15088835j, -6.25000000e-02 - 0.0625j,
              2.13388348e-01 + 0.j, -3.46944695e-17 + 0.08838835j],
             [4.77564280e-02 - 0.07437623j, -3.08076432e-02 - 0.01978136j,
              -8.63608307e-02 + 0.01882304j, 7.79675946e-03 + 0.03577183j,
              6.25000000e-02 + 0.0625j, -2.58883476e-02 + 0.02588835j,
              -3.46944695e-17 - 0.08838835j, 3.66116524e-02 + 0.j]],
            dtype=np.complex128
        )

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Density Matrix Simulator unit test finished!")

    def test_simulator(self):
        simulator = DensityMatrixSimulation()
        density_matrix = simulator.run(TestDensityMatrixSimulator.circuit)
        assert np.allclose(density_matrix, TestDensityMatrixSimulator.result)

        for _ in range(30):
            simulator = CircuitSimulator()
            _ = simulator.run(TestDensityMatrixSimulator.circuit)
            measure_res = simulator.sample(TestDensityMatrixSimulator.circuit)
            print(measure_res)
            assert 1 

    

if __name__ == "__main__":
    unittest.main()
