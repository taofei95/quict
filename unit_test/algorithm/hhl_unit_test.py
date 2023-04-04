import numpy as np
import unittest
from scipy import stats, sparse

from QuICT.algorithm.quantum_algorithm.hhl import HHL

from QuICT.simulation.state_vector import StateVectorSimulator


def random_matrix(size):
    rvs = stats.norm().rvs
    while(1):
        X = sparse.random(
            size, size, density=1, data_rvs=rvs,
            dtype=np.complex128)
        A = X.todense()
        v = np.linalg.eigvals(A)
        A = np.round(A, 3)
        if np.linalg.det(A) != 0 and np.log2(max(abs(v)) / min(abs(v))) < 6:
            return np.array(A)


def random_vector(size):
    return np.complex128(np.round(np.random.rand(size), 3) - np.full(size, 0.5))


def MSE(x, y):
    n = len(x)
    res0 = np.linalg.norm(x + y) / n
    res1 = np.linalg.norm(x - y) / n
    return min(res0, res1)


class TestHHL(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("The HHL unit test start!")
        cls.simulator = StateVectorSimulator(device="GPU")

    @classmethod
    def tearDownClass(cls) -> None:
        print("The HHL unit test finished!")

    def test_hhl_accuracy(self):
        t = [4, 8, 10, 8, 2]
        print(f"run with size 2 ** {list(range(1, 6))} for {t} times")
        for n in range(5):
            size = 1 << n + 1
            for _ in t[n]:
                matrix = random_matrix(size)
                vector = random_vector(size)

                np_slt = np.linalg.solve(matrix, vector)
                np_slt /= np.linalg.norm(np_slt)
                hhl_slt = HHL(TestHHL.simulator).run(
                    matrix, vector, measure=False)
                hhl_slt /= np.linalg.norm(hhl_slt)
                e = MSE(np_slt, hhl_slt)
                error += e
                print(f"For size = {1 << n + 1}, mean square error = {e}")

        print(f"In 32 test, mean square error = {error / 32}")
        assert 1

    def test_hhl_success_rate(self):
        for n in range(2):
            size = 1 << n + 1
            matrix = random_matrix(size)
            vector = random_vector(size)
            times = 1
            while not HHL(TestHHL.simulator).run(
                    matrix, vector):
                times += 1
            print(f"For size = {1 << n + 1}, success rate = {1.0 / times}")
        assert 1
