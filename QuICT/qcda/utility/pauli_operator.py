"""
Compute the disentangler of Pauli operators (O, O')
"""

import copy
import random
import itertools

from QuICT.core.gate import build_gate, BasicGate, CompositeGate, GateType, PAULI_GATE_SET, CX, H


class PauliOperator(object):
    """
    Pauli operator is a list of (I, X, Y, Z) with length n, which operates on n qubits.

    In this class, we use a list of int to represent the operator, where the GateTypes stand
    for the gates.
    """
    def __init__(self, operator=None, phase=1 + 0j):
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
        phase_list = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
        if phase not in phase_list:
            raise ValueError("phase must be ±1 or ±i")
        self._phase = phase

    def __str__(self):
        string = ''
        for i in range(len(self.operator)):
            if self.operator[i] == GateType.id:
                string += 'I'
            elif self.operator[i] == GateType.x:
                string += 'X'
            elif self.operator[i] == GateType.y:
                string += 'Y'
            elif self.operator[i] == GateType.z:
                string += 'Z'
            else:
                raise ValueError('Invalid GateType')
        return string + ', {}'.format(self.phase)

    @property
    def operator(self) -> list:
        return self._operator

    @operator.setter
    def operator(self, operator: list):
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
        phase_list = [1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j]
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

    @staticmethod
    def random(width: int):
        """
        Give a random PauliOperator with given width

        Args:
            width(int): the width of the PauliOperator

        Returns:
            PauliOperator: a random PauliOperator with given width
        """
        pauli_list = [GateType.id, GateType.x, GateType.y, GateType.z]
        operator = []
        for _ in range(width):
            operator.append(random.choice(pauli_list))
        return PauliOperator(operator)

    @staticmethod
    def random_anti_commutative_pair(width):
        op_id = [GateType.id for _ in range(width)]
        # Avoid p1 = identity
        while True:
            p1 = PauliOperator.random(width)
            if p1.operator != op_id:
                break
        # Expectation = 2
        while True:
            p2 = PauliOperator.random(width)
            if not p1.commute(p2):
                break
        return p1, p2

    @staticmethod
    def iterator(width: int):
        """
        Yield all the PauliOperators with given width

        Args:
            width(int): the width of the PauliOperator

        Yields:
            PauliOperator: PauliOperator with given width
        """
        pauli_list = [GateType.id, GateType.x, GateType.y, GateType.z]
        for operator in itertools.product(pauli_list, repeat=width):
            yield PauliOperator(list(operator))

    def commute(self, other):
        """
        Decide whether two PauliOperators commute or anti-commute

        Args:
            other(PauliOperator): PauliOperator to be checked with self

        Returns:
            boolean: True for commutative or False for anti-commutative
        """
        assert isinstance(other, PauliOperator),\
            TypeError("commute() only checks commutativity between PauliOperators.")
        assert self.width == other.width,\
            ValueError("PauliOperators to be checked must have the same width.")

        res = True
        for qubit in range(self.width):
            if self.operator[qubit] == GateType.id or other.operator[qubit] == GateType.id\
               or self.operator[qubit] == other.operator[qubit]:
                continue
            else:
                res = not res
        return res

    def conjugate_act(self, gate: BasicGate):
        """
        Compute the PauliOperator after conjugate action of a clifford gate
        Be aware that the conjugate action means U^-1 P U, where U is the clifford gate
        and P is the PauliOperator. It is important for S gate.

        Args:
            gate(BasicGate): the clifford gate to be acted on the PauliOperator
        """
        if not gate.is_clifford() and gate.type != GateType.id:
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
            # H Z H = X
            if self.operator[gate.targ] == GateType.z:
                self.operator[gate.targ] = GateType.x
                return
        if gate.type == GateType.s:
            assert not out_of_range(gate)
            # Sdg I S = I
            if self.operator[gate.targ] == GateType.id:
                return
            # Sdg X S = Y
            if self.operator[gate.targ] == GateType.x:
                self.operator[gate.targ] = GateType.y
                return
            # Sdg Y S = -X
            if self.operator[gate.targ] == GateType.y:
                self.operator[gate.targ] = GateType.x
                self.phase *= -1
                return
            # Sdg Z S = Z
            if self.operator[gate.targ] == GateType.z:
                return
        if gate.type == GateType.sdg:
            assert not out_of_range(gate)
            # S I Sdg = I
            if self.operator[gate.targ] == GateType.id:
                return
            # S X Sdgg = -Y
            if self.operator[gate.targ] == GateType.x:
                self.operator[gate.targ] = GateType.y
                self.phase *= -1
                return
            # S Y Sdg = X
            if self.operator[gate.targ] == GateType.y:
                self.operator[gate.targ] = GateType.x
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

    @staticmethod
    def disentangler(pauli_x, pauli_z, target=0) -> CompositeGate:
        """
        For anti-commuting n-qubit Pauli operators O and O', there exists a Clifford circuit L∈C_n
        such that L^(-1) O L = X_j, L^(-1) O' L = Z_j.
        L is referred to as a disentangler for the pair (O, O').

        Args:
            pauli_x(PauliOperator): the PauliOperator to be transformed to X_j
            pauli_z(PauliOperator): the PauliOperator to be transformed to Z_j
            target(int, optional): the j in the X_j, Z_j to be transformed to

        Returns:
            CompositeGate: the disentangler for the pair (O, O')

        Reference:
            https://arxiv.org/abs/2105.02291
        """
        assert isinstance(pauli_x, PauliOperator) and isinstance(pauli_z, PauliOperator),\
            TypeError("disentangler only defined for PauliOperator pairs")
        assert pauli_x.width == pauli_z.width,\
            ValueError('two PauliOperators must be of the same width')
        assert not pauli_x.commute(pauli_z),\
            ValueError('anti-commutative pairs needed for computing disentangler')
        assert isinstance(target, int) and 0 <= target and target < pauli_x.width,\
            ValueError('the target must be integer in the width of the PauliOperators')

        pauli_x = copy.deepcopy(pauli_x)
        pauli_z = copy.deepcopy(pauli_z)

        # Record the qubits in the 5 cases of the standard forms
        XZ = []
        XX = []
        XI = []
        IZ = []
        II = []

        def standardize(gate_type, qubit):
            gate = build_gate(gate_type, qubit)
            pauli_x.conjugate_act(gate)
            pauli_z.conjugate_act(gate)
            standardizer.append(gate)

        # Transform the pauli_x and pauli_z to 'standard' forms
        standardizer = CompositeGate()
        for qubit in range(pauli_x.width):
            if pauli_x.operator[qubit] == GateType.id:
                # II
                if pauli_z.operator[qubit] == GateType.id:
                    II.append(qubit)
                    continue
                # IX
                if pauli_z.operator[qubit] == GateType.x:
                    standardize(GateType.h, qubit)
                    IZ.append(qubit)
                    continue
                # IY
                if pauli_z.operator[qubit] == GateType.y:
                    standardize(GateType.s, qubit)
                    standardize(GateType.h, qubit)
                    IZ.append(qubit)
                    continue
                # IZ
                if pauli_z.operator[qubit] == GateType.z:
                    IZ.append(qubit)
                    continue
            if pauli_x.operator[qubit] == GateType.x:
                # XI
                if pauli_z.operator[qubit] == GateType.id:
                    XI.append(qubit)
                    continue
                # XX
                if pauli_z.operator[qubit] == GateType.x:
                    XX.append(qubit)
                    continue
                # XY
                if pauli_z.operator[qubit] == GateType.y:
                    standardize(GateType.h, qubit)
                    standardize(GateType.s, qubit)
                    standardize(GateType.h, qubit)
                    XZ.append(qubit)
                    continue
                # XZ
                if pauli_z.operator[qubit] == GateType.z:
                    XZ.append(qubit)
                    continue
            if pauli_x.operator[qubit] == GateType.y:
                # YI
                if pauli_z.operator[qubit] == GateType.id:
                    standardize(GateType.s, qubit)
                    XI.append(qubit)
                    continue
                # YX
                if pauli_z.operator[qubit] == GateType.x:
                    standardize(GateType.h, qubit)
                    standardize(GateType.s, qubit)
                    XZ.append(qubit)
                    continue
                # YY
                if pauli_z.operator[qubit] == GateType.y:
                    standardize(GateType.s, qubit)
                    XX.append(qubit)
                    continue
                # YZ
                if pauli_z.operator[qubit] == GateType.z:
                    standardize(GateType.s, qubit)
                    XZ.append(qubit)
                    continue
            if pauli_x.operator[qubit] == GateType.z:
                # ZI
                if pauli_z.operator[qubit] == GateType.id:
                    standardize(GateType.h, qubit)
                    XI.append(qubit)
                    continue
                # ZX
                if pauli_z.operator[qubit] == GateType.x:
                    standardize(GateType.h, qubit)
                    XZ.append(qubit)
                    continue
                # ZY
                if pauli_z.operator[qubit] == GateType.y:
                    standardize(GateType.s, qubit)
                    standardize(GateType.h, qubit)
                    XZ.append(qubit)
                    continue
                # ZZ
                if pauli_z.operator[qubit] == GateType.z:
                    standardize(GateType.h, qubit)
                    XX.append(qubit)
                    continue

        # Some tests for standardizer
        assert len(XZ + XX + XI + IZ + II) == pauli_x.width
        for qubit in XZ:
            assert pauli_x.operator[qubit] == GateType.x and pauli_z.operator[qubit] == GateType.z
        for qubit in XX:
            assert pauli_x.operator[qubit] == GateType.x and pauli_z.operator[qubit] == GateType.x
        for qubit in XI:
            assert pauli_x.operator[qubit] == GateType.x and pauli_z.operator[qubit] == GateType.id
        for qubit in IZ:
            assert pauli_x.operator[qubit] == GateType.id and pauli_z.operator[qubit] == GateType.z
        for qubit in II:
            assert pauli_x.operator[qubit] == GateType.id and pauli_z.operator[qubit] == GateType.id

        # Construct the disentangler of 'standard' pairs with the algorithm given in reference
        disentangler = CompositeGate()
        with disentangler:
            for subset in [XZ, XX, XI, IZ, II]:
                if target in subset:
                    if subset is not XZ:
                        # Swap & [target, XZ[0]]
                        CX & [target, XZ[0]]
                        CX & [XZ[0], target]
                        CX & [target, XZ[0]]
                    if XZ[0] != target:
                        subset.remove(target)
                        subset.append(XZ[0])
                        XZ[0] = target
            for j in XI:
                CX & [target, j]
            for j in IZ:
                CX & [j, target]
            if len(XX) > 0:
                i = XX[0]
                for j in XX[1:]:
                    CX & [i, j]
                CX & [target, i]
                H & i
                CX & [i, target]
            # Anti-commutation gives that len(XZ) is odd
            for k in range((len(XZ) - 1) // 2):
                i = 2 * k + 1
                j = 2 * k + 2
                CX & [XZ[j], XZ[i]]
                CX & [XZ[i], target]
                CX & [target, XZ[j]]

        standardizer.extend(disentangler)

        # Add the last Pauli gate for the phase correction
        for gate in disentangler:
            pauli_x.conjugate_act(gate)
            pauli_z.conjugate_act(gate)
        if pauli_x.phase == 1 and pauli_z.phase == -1:
            gate = build_gate(GateType.x, target)
            standardizer.append(gate)
        if pauli_x.phase == -1 and pauli_z.phase == -1:
            gate = build_gate(GateType.y, target)
            standardizer.append(gate)
        if pauli_x.phase == -1 and pauli_z.phase == 1:
            gate = build_gate(GateType.z, target)
            standardizer.append(gate)

        return standardizer
