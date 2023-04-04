import cupy as cp
import numpy as np

from QuICT.core.circuit import Circuit
from QuICT.simulation.utils import GateSimulator


class Differentiator:
    def __init__(
        self,
        device: str = "GPU",
        precision: str = "double",
        gpu_device_id: int = 0,
        sync: bool = True,
    ):
        self._simulator = GateSimulator(device, gpu_device_id, sync)

    def __call__(
        self, circuit: Circuit, state_vector: np.ndarray, params, expectation_op
    ):
        raise NotImplementedError
