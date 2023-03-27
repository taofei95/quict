import cupy as cp
import numpy as np

from QuICT.core.circuit import Circuit
from QuICT.simulation.state_vector import StateVectorSimulator


class Differentiator:
    def __init__(self, simulator: StateVectorSimulator):
        self._simulator = simulator

    def __call__(
        self, circuit: Circuit, state_vector: np.ndarray, params, expectation_op
    ):
        raise NotImplementedError
