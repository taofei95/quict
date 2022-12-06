import unittest
import numpy as np

from QuICT.algorithm.quantum_algorithm.hhl import LinearEquation


class TestHHl(unittest.TestCase):

    def test(self):
        # A = np.array([[7, 0, 0, 0],
        #               [0, 3, 0, 0],
        #               [0, 0, 3, 0],
        #               [0, 0, 0, 4]])
        # b = np.array([1, 1, 1, 1])
        measure = False

        A = np.array([[3., -1 / 5],
                      [-1 / 5, 2.]])
        b = np.array([1, 1])

        test = LinearEquation(A, b)
        test.test(measure=measure)


if __name__ == "__main__":
    unittest.main()
