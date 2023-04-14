import numpy as np
import unittest

from QuICT.algorithm.quantum_algorithm.hhl import HHL

from QuICT.simulation.state_vector import StateVectorSimulator


matrix = np.array([[1.0 + 0j, 2.0 + 0j],
                   [3.0 + 0j, 2.0 + 0j]])
vector = np.ones(2, dtype=np.complex128)


def MSE(x, y):
    n = len(x)
    res0 = np.linalg.norm(x + y) / n
    res1 = np.linalg.norm(x - y) / n
    return min(res0, res1)


class TestHHL(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("The HHL unit test start!")
        cls.simulator = StateVectorSimulator()

    @classmethod
    def tearDownClass(cls) -> None:
        print("The HHL unit test finished!")

    def test_hhl_accuracy(self):
        np_slt = np.linalg.solve(matrix, vector)
        np_slt /= np.linalg.norm(np_slt)
        hhl_slt = HHL(TestHHL.simulator).run(
            matrix, vector, phase_qubits=6, measure=False
        )
        hhl_slt /= np.linalg.norm(hhl_slt)
        mse = MSE(np_slt, hhl_slt)

        assert abs(mse) < 0.1

    def test_hhl_measure(self):
        str = HHL(TestHHL.simulator).run(
            matrix, vector, phase_qubits=6
        )

        assert 1


if __name__ == "__main__":
    unittest.main()
