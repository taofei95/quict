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

    def __init__(self, gate, noise):
        self._gate = gate
        self._noise = noise
        args_num = gate.controls + gate.targets
        gate_name = gate.type.name
        super().__init__(args_num, name=f"ng_{gate_name}")
        self._noise_matrix = noise.apply_to_gate(gate.matrix)
        self._precision = gate.precision

    def copy(self):
        _ngate = NoiseGate(self._gate, self._noise)

        if len(self.targs) > 0:
            _ngate.targs = self._targs

        return _ngate
