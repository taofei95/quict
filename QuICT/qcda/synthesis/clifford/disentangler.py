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
    def gates(self) -> CompositeGate:
        """
        The CompositeGate corresponding to the PauliOperator

        Returns:
            CompositeGate: The CompositeGate corresponding to the PauliOperator
        """
        gates = CompositeGate()
        for qubit, gatetype in enumerate(self.operator):
            gate = build_gate(gatetype, qubit)
            gates.append(gate)
        return gates

    @property
    def width(self) -> int:
        return len(self.operator)

    def conjugate_act(self, gate: BasicGate):
        """
        Compute the PauliOperator after conjugate action of a clifford gate
        Be aware that the conjugate action means U P U^-1, where U is the clifford gate
        and P is the PauliOperator. It is important for S gate.

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
            if self.operator[gate.carg] == GateType.id:
                # CX I0 I1 CX = I0 I1
                if self.operator[gate.targ] == GateType.id:
                    return
                # CX I0 X1 CX = I0 X1
                if self.operator[gate.targ] == GateType.x:
                    return
                # CX I0 Y1 CX = Z0 Y1
                if self.operator[gate.targ] == GateType.y:
                    self.operator[gate.carg] = GateType.z
                    self.operator[gate.targ] = GateType.y
                    return
                # CX I0 Z1 CX = Z0 Z1
                if self.operator[gate.targ] == GateType.z:
                    self.operator[gate.carg] = GateType.z
                    return
            if self.operator[gate.carg] == GateType.x:
                # CX X0 I1 CX = X0 X1
                if self.operator[gate.targ] == GateType.id:
                    self.operator[gate.targ] = GateType.x
                    return
                # CX X0 X1 CX = X0 I1
                if self.operator[gate.targ] == GateType.x:
                    self.operator[gate.targ] = GateType.id
                    return
                # CX X0 Y1 CX = Y0 Z1
                if self.operator[gate.targ] == GateType.y:
                    self.operator[gate.carg] = GateType.y
                    self.operator[gate.targ] = GateType.z
                    return
                # CX X0 Z1 CX = -Y0 Y1
                if self.operator[gate.targ] == GateType.z:
                    self.operator[gate.carg] = GateType.y
                    self.operator[gate.targ] = GateType.y
                    self.phase *= -1
                    return
            if self.operator[gate.carg] == GateType.y:
                # CX Y0 I1 CX = Y0 X1
                if self.operator[gate.targ] == GateType.id:
                    self.operator[gate.targ] = GateType.x
                    return
                # CX Y0 X1 CX = Y0 I1
                if self.operator[gate.targ] == GateType.x:
                    self.operator[gate.targ] = GateType.id
                    return
                # CX Y0 Y1 CX = -X0 Z1
                if self.operator[gate.targ] == GateType.y:
                    self.operator[gate.carg] = GateType.x
                    self.operator[gate.targ] = GateType.z
                    self.phase *= -1
                    return
                # CX Y0 Z1 CX = X0 Y1
                if self.operator[gate.targ] == GateType.z:
                    self.operator[gate.carg] = GateType.x
                    self.operator[gate.targ] = GateType.y
                    return
            if self.operator[gate.carg] == GateType.z:
                # CX Z0 I1 CX = Z0 I1
                if self.operator[gate.targ] == GateType.id:
                    return
                # CX Z0 X1 CX = Z0 X1
                if self.operator[gate.targ] == GateType.x:
                    return
                # CX Z0 Y1 CX = I0 Y1
                if self.operator[gate.targ] == GateType.y:
                    self.operator[gate.carg] = GateType.id
                    return
                # CX Z0 Z1 CX = I0 Z1
                if self.operator[gate.targ] == GateType.z:
                    self.operator[gate.carg] = GateType.id
                    return
        if gate.type == GateType.h:
            assert not out_of_range(gate)
            # H I H = I
            if self.operator[gate.targ] == GateType.id:
                return
            # H X H = Z
            if self.operator[gate.targ] == GateType.x:
                self.operator[gate.targ] = GateType.z
                return
            # H Y H = -Y
            if self.operator[gate.targ] == GateType.y:
                self.phase *= -1
                return
            # H Z H = Z
            if self.operator[gate.targ] == GateType.z:
                self.operator[gate.targ] = GateType.x
                return
        if gate.type == GateType.s:
            assert not out_of_range(gate)
            # S I Sdg = I
            if self.operator[gate.targ] == GateType.id:
                return
            # S X Sdg = Y
            if self.operator[gate.targ] == GateType.x:
                self.operator[gate.targ] = GateType.y
                return
            # S Y Sdg = -X
            if self.operator[gate.targ] == GateType.y:
                self.operator[gate.targ] = GateType.x
                self.phase *= -1
                return
            # S Z Sdg = Z
            if self.operator[gate.targ] == GateType.z:
                return
        if gate.type == GateType.x:
            assert not out_of_range(gate)
            # X I X = I
            if self.operator[gate.targ] == GateType.id:
                return
            # X X X = X
            if self.operator[gate.targ] == GateType.x:
                return
            # X Y X = -Y
            if self.operator[gate.targ] == GateType.y:
                self.phase *= -1
                return
            # X Z X = -Z
            if self.operator[gate.targ] == GateType.z:
                self.phase *= -1
                return
        if gate.type == GateType.y:
            assert not out_of_range(gate)
            # Y I Y = I
            if self.operator[gate.targ] == GateType.id:
                return
            # Y X Y = -X
            if self.operator[gate.targ] == GateType.x:
                self.phase *= -1
                return
            # Y Y Y = Y
            if self.operator[gate.targ] == GateType.y:
                return
            # Y Z Y = -Z
            if self.operator[gate.targ] == GateType.z:
                self.phase *= -1
                return
        if gate.type == GateType.z:
            assert not out_of_range(gate)
            # Z I Z = I
            if self.operator[gate.targ] == GateType.id:
                return
            # Z X Z = -X
            if self.operator[gate.targ] == GateType.x:
                self.phase *= -1
                return
            # Z Y Z = -Y
            if self.operator[gate.targ] == GateType.y:
                self.phase *= -1
                return
            # Z Z Z = Z
            if self.operator[gate.targ] == GateType.z:
                return


class PauliDisentangler(object):
    """
    For anti-commuting n-qubit Pauli operators O and O', there exists a Clifford circuit L∈C_n
    such that L^(-1) O L = X_1, L^(-1) O' L = Z_1.
    L is referred to as a disentangler for the pair (O, O').

    Reference:
        https://arxiv.org/abs/2105.02291
    """
