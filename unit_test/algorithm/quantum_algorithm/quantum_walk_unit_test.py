import numpy as np
import unittest

from QuICT.core.gate import H
from QuICT.algorithm.quantum_algorithm.quantum_walk import QuantumWalk, Graph
from QuICT.simulation.cpu_simulator import CircuitSimulator
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator


class TestQuantumWalk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("The Random Walk unit test start!")
        cls.simulator = ConstantStateVectorSimulator()
        cls.steps = 10

    @classmethod
    def tearDownClass(cls) -> None:
        print("The Random Walk unit test finished!")

    def test_circular_random_walk(self):
        edges = [[3, 1], [0, 2], [1, 3], [2, 0]]
        graph = Graph(4, edges)

        qw = QuantumWalk(TestQuantumWalk.steps, graph, H.matrix)
        _ = qw.run(TestQuantumWalk.simulator)

        assert 1

    def test_unbalanced_random_walk(self):
        edges = [[2, 1], [0, 2], [1, 0]]
        graph = Graph(3, edges)

        qw = QuantumWalk(TestQuantumWalk.steps, graph, H.matrix)
        _ = qw.run(TestQuantumWalk.simulator)

        assert 1

    def test_2qcoin_random_walk(self):
        edges = [list(np.random.choice(8, size=4, replace=False)) for _ in range(8)]
        graph = Graph(8, edges)

        qw = QuantumWalk(3, graph, np.kron(H.matrix, H.matrix))
        _ = qw.run(TestQuantumWalk.simulator)

        assert 1


if __name__ == "__main__":
    unittest.main()
