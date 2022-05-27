from QuICT.core.gate import BasicGate
from ._operator import Operator


class NoiseGate(Operator):
    """
    The quantum gate with noise error.
    """
    @property
    def gate_matrix(self):
        return self._gate.matrix

    @property
    def gate_type(self):
        return self._gate.type

    @property
    def noise_matrix(self):
        return self._noise_matrix

    @property
    def noise_type(self):
        return self._error.type

    @property
    def kraus_operators(self):
        return self._error.kraus_operators

    @property
    def kraus_operators_ctranspose(self):
        return self._error.kraus_operators_ctranspose

    def __init__(self, gate: BasicGate, error):
        """
        Args:
            gate (BasicGate): The quantum gate.
            error (QuantumNoiseError): The noise error.
        """
        assert isinstance(gate, BasicGate)
        super().__init__(gate.controls + gate.targets)
        self.targ = gate.cargs + gate.targs

        self._gate = gate
        self._error = error
        self._noise_matrix = self._error.apply_to_gate(gate.matrix)
