import numpy as np
import unittest

from QuICT.algorithm.quantum_algorithm.hhl import HHL

from QuICT.simulation.state_vector import StateVectorSimulator
from example.algorithm.random_sparse import RS


class TestHHL(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        print("The HHL unit test start!")
        cls.simulator = StateVectorSimulator(device="GPU")

    @classmethod
    def tearDownClass(cls) -> None:
        print("The HHL unit test finished!")

    def test_hhl_without_measure(self):
        pass

    def test_hhl_success_times(self):
        pass
