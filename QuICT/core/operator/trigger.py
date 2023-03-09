from typing import Tuple, Union, List, Dict
from types import FunctionType

from ._operator import Operator
from QuICT.tools.exception.core import TypeError, ValueError


class Trigger(Operator):
    """
    The trigger for switch the dynamic circuit; contains the target qubits and
    related circuits with different state.
    """
    def __init__(
        self,
        targets: int,
        state_gate_mapping: Union[dict, list, tuple, FunctionType]
    ):
        """
        Args:
            targets (int): The number of target qubits.
            state_gate_mapping: The mapping of state and related composite gates.
                (Union[Dict[int, CompositeGate], List[CompositeGate], Tuple[CompositeGate], FunctionType])

        Raises:
            TypeError: Error input parameters.
        """
        super().__init__(targets=targets)

        # Deal with state - compositegate mapping
        self._state_gate_mapping = {}
        if isinstance(state_gate_mapping, (list, tuple, dict, FunctionType)):
            raise TypeError("Trigger.state_gate_mapping", "[list, tuple, dict, Function]", {type(state_gate_mapping)})

    def mapping(self, state: int):
        """ Return the related composite gate with given qubits' measured state.

        Args:
            state (int): The qubits' measured state.

        Returns:
            CompositeGate: The related composite gate.
        """
        assert state >= 0 and state < 2 ** self.targets, ValueError(
            "Trigger.mapping.state", f"[0, {2**self.targets}]", state
        )

        if isinstance(self._state_gate_mapping, FunctionType):
            return self._state_gate_mapping(state)
        else:
            return self._state_gate_mapping[state]
