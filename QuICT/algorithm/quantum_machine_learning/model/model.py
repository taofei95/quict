from abc import ABC, abstractmethod
import numpy as np
from numpy_ml.neural_nets.optimizers import *
from typing import List, Union

from QuICT.algorithm.quantum_machine_learning.differentiator import Differentiator
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.simulation.state_vector import StateVectorSimulator


class Model(ABC):
    """The abstract class for model."""

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    def __init__(
        self,
        n_qubits: int,
        optimizer: OptimizerBase,
        hamiltonian: Union[Hamiltonian, List] = None,
        params: np.ndarray = None,
        device: str = "GPU",
        gpu_device_id: int = 0,
        differentiator: str = "adjoint",
    ):
        self._n_qubits = n_qubits
        self._optimizer = optimizer
        self._params = params
        self._hamiltonian = hamiltonian
        self._simulator = StateVectorSimulator(
            device=device, gpu_device_id=gpu_device_id
        )
        self._differentiator = Differentiator(
            device=device, backend=differentiator, gpu_device_id=gpu_device_id
        )

    @abstractmethod
    def run_step():
        raise NotImplementedError

    @abstractmethod
    def update():
        raise NotImplementedError
