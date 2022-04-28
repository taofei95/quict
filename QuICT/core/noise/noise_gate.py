import copy

from QuICT.core.gate import BasicGate
from .noise_error import QuantumNoiseError


class NoiseGate(BasicGate):
    """
    The quantum gate with noise error.
    """
    @property
    def noise_matrix(self):
        return self._error.apply_to_gate(self.matrix)

    @property
    def noise_type(self):
        return self._error.type

    @property
    def kraus_operators(self):
        return self._error.kraus_operators

    @property
    def kraus_operators_ctranspose(self):
        return self._error.kraus_operators_ctranspose

    def __init__(self, gate: BasicGate, error: QuantumNoiseError):
        """
        Args:
            gate (BasicGate): The quantum gate.
            error (QuantumNoiseError): The noise error.
        """
        assert isinstance(gate, BasicGate) and isinstance(error, QuantumNoiseError)
        super().__init__(
            gate.controls,
            gate.targets,
            gate.params,
            gate.type
        )

        self._gate_args_copy(gate)
        self._error = error

    def _gate_args_copy(self, gate: BasicGate):
        """ Copy the qubit args from the given gate.

        Args:
            gate (BasicGate): The quantum gate.
        """
        if gate.cargs:
            self.cargs = copy.deepcopy(gate.cargs)

        if gate.targs:
            self.targs = copy.deepcopy(gate.targs)

        if gate.pargs:
            self.pargs = copy.deepcopy(gate.pargs)

        if gate.assigned_qubits:
            self.assigned_qubits = copy.deepcopy(gate.assigned_qubits)
            self.update_name(gate.assigned_qubits[0].id)
