from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from QuICT.core.gate import Variable


class Ansatz(ABC):
    """The abstract class for ansatz.

    Args:
        n_qubits (int): The number of qubits.
    """

    @property
    def params(self) -> Variable:
        """Get the parameters.

        Returns:
            Variable: The parameters.
        """
        return self._params

    def __init__(self, n_qubits: int):
        """Initialize an Ansatz object."""
        self._n_qubits = n_qubits
        self._params = None

    @abstractmethod
    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        """Initialize an ansatz with trainable parameters."""
        raise NotImplementedError
