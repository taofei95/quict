from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
from numpy_ml.neural_nets.optimizers import *

from QuICT.algorithm.quantum_machine_learning.differentiator import Differentiator
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.simulation.state_vector import StateVectorSimulator


class Model(ABC):
    """Base class for all models involving variations.

    Note:
        User-defined models also need to inherit this class.

    Args:
        n_qubits (int): The number of qubits.
        optimizer (OptimizerBase): The optimizer used to optimize the network.
        hamiltonian (Union[Hamiltonian, List], optional): The hamiltonians for measurement. Defaults to None.
        params (np.ndarray, optional): Initialization parameters. Defaults to None.
        device (str, optional): The device type, one of [CPU, GPU]. Defaults to "CPU".
        gpu_device_id (int, optional): The GPU device ID. Defaults to 0.
        differentiator (str, optional): The differentiator type. Defaults to "adjoint".
    """

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = params

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def __init__(
        self,
        n_qubits: int,
        optimizer: OptimizerBase,
        hamiltonian: Union[Hamiltonian, List] = None,
        params: np.ndarray = None,
        device: str = "CPU",
        gpu_device_id: int = 0,
        differentiator: str = "adjoint",
    ):
        """Initialize a Model instance."""
        self._n_qubits = n_qubits
        self._optimizer = optimizer
        self._params = params
        self._params_grads = None
        self._hamiltonian = hamiltonian
        self._simulator = StateVectorSimulator(
            device=device, gpu_device_id=gpu_device_id
        )
        self._differentiator = Differentiator(
            device=device, backend=differentiator, gpu_device_id=gpu_device_id
        )

    @abstractmethod
    def forward(self):
        """The forward propagation procedure for one step."""
        raise NotImplementedError

    @abstractmethod
    def backward(self):
        """The backward propagation procedure for one step."""
        raise NotImplementedError

    @abstractmethod
    def update(self):
        """Update the trainable parameters in the PQC."""
        raise NotImplementedError
