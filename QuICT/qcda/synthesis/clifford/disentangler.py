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
    def __init__(self, operator=None):
        """
        Construct a PauliOperator with a list of int(GateType)

        Args:
            operator(list, optional): the list of int(GateType) representing the PauliOperator
        """
        if operator is None:
            self.operator = []
        else:
            if not isinstance(operator, list):
                raise TypeError("operator must be list of int(GateType).")
            for gatetype in operator:
                if gatetype != GateType.id and gatetype not in PAULI_GATE_SET:
                    raise ValueError("operator must contain Pauli gates only.")
            self.operator = operator

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

    def conjugate_act(self, gate: BasicGate):
        """
        Compute the PauliOperator after conjugate action of a clifford gate

        Args:
            gate(BasicGate): the clifford gate to be acted on the PauliOperator
        """
        if not gate.is_clifford(): 
            raise TypeError("Only conjugate action of Clifford gates here.")


class PauliDisentangler(object):
    """
    For anti-commuting n-qubit Pauli operators O and O', there exists a Clifford circuit L∈C_n
    such that L^(-1) O L = X_1, L^(-1) O' L = Z_1.
    L is referred to as a disentangler for the pair (O, O').

    Reference:
        https://arxiv.org/abs/2105.02291
    """
