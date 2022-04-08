import copy
from typing import Tuple, Union, List, Dict
from types import FunctionType

from QuICT.core.gate import CompositeGate, BasicGate
from QuICT.core.noise import QuantumNoiseError


class Trigger:
    """
    The trigger for switch the dynamic circuit; contains the target qubits and
    related circuits with different state.
    """
    @property
    def targets(self):
        return self._targets

    @property
    def targs(self):
        return self._targs

    @targs.setter
    def targs(self, targets):
        assert isinstance(targets, (int, list))
        self._targs = targets

    @property
    def targ(self):
        return self._targs[0]

    def __init__(
        self,
        targets: int,
        state_gate_mapping: Union[Dict[int, CompositeGate], List[CompositeGate], Tuple[CompositeGate], FunctionType]
    ):
        """
        Args:
            targets (int): The number of target qubits.
            state_gate_mapping: The mapping of state and related composite gates.
                (Union[Dict[int, CompositeGate], List[CompositeGate], Tuple[CompositeGate], FunctionType])

        Raises:
            TypeError: Error input parameters.
        """
        assert targets >= 0 and isinstance(targets, int), f"targets must be a positive integer, not {type(targets)}"
        self._targets = targets
        self._targs = []

        self._state_gate_mapping = {}
        if isinstance(state_gate_mapping, (list, tuple)):
            for idx, cgate in enumerate(state_gate_mapping):
                assert isinstance(cgate, (CompositeGate, BasicGate)), \
                    "Only accept CompositeGate or BasicGate for state_gate_mapping."
                self._state_gate_mapping[idx] = cgate
        elif isinstance(state_gate_mapping, dict):
            for key, value in state_gate_mapping.items():
                assert isinstance(key, int) and isinstance(value, (CompositeGate, BasicGate))

            self._state_gate_mapping = state_gate_mapping
        elif isinstance(state_gate_mapping, FunctionType):
            self._check_function_validation(state_gate_mapping)
            self._state_gate_mapping = state_gate_mapping
        else:
            raise TypeError(
                f"The trigger's mapping should be one of [list, dict, Function], not {type(state_gate_mapping)}."
            )

    def __and__(self, targets: Union[List[int], int],):
        """ Assigned the trigger's target qubits.

        Args:
            targets (Union[List[int], int]): The indexes of target qubits.
        """
        if isinstance(targets, int):
            assert targets >= 0
            self._targs = [targets]
        elif isinstance(targets, list):
            for q in targets:
                assert targets >= 0 and isinstance(q, int), f"targets must be a positive integer, not {q}"
            self._targs = targets
        else:
            raise TypeError(f"qubits must be one of [List[int], int], not {type(targets)}")

    def __or__(self, targets):
        """ Append the trigger to the circuit.

        Args:
            targets (Circuit): The circuit which contains the trigger.
        """
        try:
            targets.append(self)
        except Exception as e:
            raise TypeError(f"Failure to append Trigger to targets, due to {e}")

    def mapping(self, state: int) -> CompositeGate:
        """ Return the related composite gate with given qubits' measured state.

        Args:
            state (int): The qubits' measured state.

        Returns:
            CompositeGate: The related composite gate.
        """
        assert state >= 0 and state < 2 ** self.targets, f"The state should between 0 and {2**self.targets}."

        if isinstance(self._state_gate_mapping, FunctionType):
            return self._state_gate_mapping(state)
        else:
            return self._state_gate_mapping[state]

    def _check_function_validation(self, state_gate_mapping):
        for i in range(2 ** self.targets):
            if not isinstance(state_gate_mapping(i), (CompositeGate, BasicGate, None)):
                raise KeyError("The trigger's mapping should only return CompositeGate for all possible state.")


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
