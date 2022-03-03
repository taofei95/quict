"""
Compute the disentangler of Pauli operators (O, O')
"""

from QuICT.core.gate import build_gate, BasicGate, CompositeGate, GateType, PAULI_GATE_SET

class PauliOperator(object):
    """
    Pauli operator is a list of (I, X, Y, Z) with length n, which operates on n qubits.

    In this class, we use a list of int to represent the operator, where the GateTypes stand
    for the gates. 
    """
    def __init__(self, operator=None, phase=1+0j):
        """
        Construct a PauliOperator with a list of int(GateType)

        Args:
            operator(list, optional): the list of int(GateType) representing the PauliOperator
            phase(complex, optional): the global phase of the PauliOperator, ±1 or ±i
        """
        if operator is None:
            self._operator = []
        else:
            if not isinstance(operator, list):
                raise TypeError("operator must be list of int(GateType).")
            for gatetype in operator:
                if gatetype != GateType.id and gatetype not in PAULI_GATE_SET:
                    raise ValueError("operator must contain Pauli gates only.")
            self._operator = operator
        phase_list = [1+0j, 1j, -1+0j, -1j]
        if phase not in phase_list:
            raise ValueError("phase must be ±1 or ±i")
        self._phase = phase

    @property
    def operator(self) -> list:
        return self._operator

    @operator.setter
    def operator(self, operator : list):
        if not isinstance(operator, list):
            raise TypeError("operator must be list of int(GateType).")
        for gatetype in operator:
            if gatetype != GateType.id and gatetype not in PAULI_GATE_SET:
                raise ValueError("operator must contain Pauli gates only.")
        self._operator = operator

    @property
    def phase(self) -> complex:
        return self._phase

    @phase.setter
    def phase(self, phase: complex):
        phase_list = [1+0j, 1j, -1+0j, -1j]
        if phase not in phase_list:
            raise ValueError("phase must be ±1 or ±i")
        self._phase = phase

    @property
    def gates(self):
        """
        The CompositeGate corresponding to the PauliOperator

        Returns:
            CompositeGate: The CompositeGate corresponding to the PauliOperator
        """
        gates = CompositeGate()
        for qubit, gatetype in enumerate(self.operator):
            if gatetype == GateType.id:
                continue
            else:
                gate = build_gate(gatetype, qubit)
                gates.append(gate)
        return gates

    @property
    def width(self):
        return len(self.operator)

    def conjugate_act(self, gate: BasicGate):
        """
        Compute the PauliOperator after conjugate action of a clifford gate

        Args:
            gate(BasicGate): the clifford gate to be acted on the PauliOperator
        """
        if not gate.is_clifford(): 
            raise ValueError("Only conjugate action of Clifford gates here.")

        def out_of_range(gate):
            targs = gate.cargs + gate.targs
            for targ in targs:
                assert targ < self.width, ValueError("target of the gate out of range")

        if gate.type == GateType.cx:
            assert not out_of_range(gate)
        elif gate.type == GateType.h:
            assert not out_of_range(gate)
        elif gate.type == GateType.s:
            assert not out_of_range(gate)
        elif gate.type == GateType.x:
            assert not out_of_range(gate)
        elif gate.type == GateType.y:
            assert not out_of_range(gate)
        elif gate.type == GateType.z:
            assert not out_of_range(gate)
        else:
            raise ValueError("Only conjugate action of Clifford gates here.")


class PauliDisentangler(object):
    """
    For anti-commuting n-qubit Pauli operators O and O', there exists a Clifford circuit L∈C_n
    such that L^(-1) O L = X_1, L^(-1) O' L = Z_1.
    L is referred to as a disentangler for the pair (O, O').

    Reference:
        https://arxiv.org/abs/2105.02291
    """
