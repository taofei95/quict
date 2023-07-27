from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from QuICT.core.gate import Variable


class Ansatz(ABC):
    """The abstract class for ansatz."""

    @property
    def params(self):
        return self._params

    def __init__(self, n_qubits: int):
        self._n_qubits = n_qubits
        self._params = None

    @abstractmethod
    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        raise NotImplementedError
