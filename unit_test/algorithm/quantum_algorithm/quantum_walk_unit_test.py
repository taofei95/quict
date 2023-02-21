import numpy as np
import unittest

from QuICT.core.gate import H
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.quantum_algorithm.quantum_walk import QuantumWalk, Graph


class TestQuantumWalk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Random Walk unit test start!")
        cls.steps = 10
        cls.simulator = StateVectorSimulator()

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Random Walk unit test finished!")

    def test_circular_random_walk(self):
        edges = [[3, 1], [0, 2], [1, 3], [2, 0]]
        qw = QuantumWalk(TestQuantumWalk.simulator)
        _ = qw.run(step=TestQuantumWalk.steps, position=4, edges=edges, coin_operator=H.matrix)

        assert 1

    def test_unbalanced_random_walk(self):
        edges = [[2, 1], [0, 2], [1, 0]]
        qw = QuantumWalk(TestQuantumWalk.simulator)
        _ = qw.run(step=TestQuantumWalk.steps, position=3, edges=edges, coin_operator=H.matrix)

        assert 1

    def test_2qcoin_random_walk(self):
        edges = [list(np.random.choice(8, size=4, replace=False)) for _ in range(8)]
        qw = QuantumWalk(TestQuantumWalk.simulator)
        _ = qw.run(step=3, position=8, edges=edges, coin_operator=np.kron(H.matrix, H.matrix))

        assert 1


if __name__ == "__main__":
    unittest.main()
