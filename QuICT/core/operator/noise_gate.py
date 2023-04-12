from __future__ import annotations

from ._operator import Operator


class NoiseGate(Operator):
    """
    The quantum gate with noise error.

    Args:
        gate (BasicGate): The quantum gate.
        error (QuantumNoiseError): The noise error.
    """
    @property
    def noise_matrix(self) -> list:
        """ The noised gate matrix. """
        return self._noise_matrix

    @property
    def qasm_name(self) -> str:
        return "noise"

    @property
    def type(self) -> str:
        return "noise"

    def __init__(self, noise_matrix, args_num: int):
        super().__init__(args_num)
        self._noise_matrix = noise_matrix
        self._precision = self._noise_matrix[0].dtype

    def copy(self):
        _ngate = NoiseGate(self._noise_matrix, self._targets)

        if len(self.targs) > 0:
            _ngate.targs = self._targs

        return _ngate
