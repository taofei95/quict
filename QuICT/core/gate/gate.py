#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/17 9:04
# @Author  : Han Yu, Li Kaiqi
# @File    : gate.py
import numpy as np
import copy

from QuICT.core.gate.composite_gate import CGATE_LIST
from QuICT.core.utils import GateType, SPECIAL_GATE_SET, DIAGONAL_GATE_SET


class BasicGate(object):
    """ the abstract SuperClass of all basic quantum gate

    All basic quantum gate described in the framework have
    some common attributes and some common functions
    which defined in this class

    Attributes:
        name(str): the name of the gate
        controls(int): the number of the control bits of the gate
        cargs(list<int>): the list of the index of control bits in the circuit
        carg(int, read only): the first object of cargs

        targets(int): the number of the target bits of the gate
        targs(list<int>): the list of the index of target bits in the circuit
        targ(int, read only): the first object of targs

        params(list): the number of the parameter of the gate
        pargs(list): the list of the parameter
        parg(read only): the first object of pargs

        qasm_name(str, read only): gate's name in the OpenQASM 2.0
        type(GateType, read only): gate's type described by GateType

        matrix(np.array): the unitary matrix of the quantum gate act on targets
    """
    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix) -> np.ndarray:
        self._matrix = matrix

    @property
    def target_matrix(self) -> np.ndarray:
        return self._matrix

    @property
    def type(self):
        return self._type

    @property
    def controls(self) -> int:
        return self._controls

    @controls.setter
    def controls(self, controls: int):
        assert isinstance(controls, int)
        self._controls = controls

    @property
    def cargs(self):
        return self._cargs

    @cargs.setter
    def cargs(self, cargs: list):
        if isinstance(cargs, list):
            self._cargs = cargs
        else:
            self._cargs = [cargs]

    @property
    def targets(self) -> int:
        return self._targets

    @targets.setter
    def targets(self, targets: int):
        assert isinstance(targets, int)
        self._targets = targets

    @property
    def targs(self):
        return self._targs

    @targs.setter
    def targs(self, targs: list):
        if isinstance(targs, list):
            self._targs = targs
        else:
            self._targs = [targs]

    @property
    def params(self) -> int:
        return self._params

    @params.setter
    def params(self, params: int):
        self._params = params

    @property
    def pargs(self):
        return self._pargs

    @pargs.setter
    def pargs(self, pargs: list):
        if isinstance(pargs, list):
            self._pargs = pargs
        else:
            self._pargs = [pargs]

        assert len(self._pargs) == self.params

    @property
    def parg(self):
        return self.pargs[0]

    @property
    def carg(self):
        return self.cargs[0]

    @property
    def targ(self):
        return self.targs[0]

    @property
    def qasm_name(self):
        return self._qasm_name

    def __init__(
        self,
        controls: int,
        targets: int,
        params: int,
        type: GateType
    ):
        self._matrix = None

        self._controls = controls
        self._targets = targets
        self._params = params
        self._cargs = []    # list of int
        self._targs = []    # list of int
        self._pargs = []    # list of float/..

        assert isinstance(type, GateType)
        self._type = type
        self._qasm_name = str(type.name)
        self._name = "-".join([str(type), "", ""])

        self.assigned_qubits = []   # list of qubits

    def __or__(self, targets):
        """deal the operator '|'

        Use the syntax "gate | circuit" or "gate | Composite Gate"
        to add the gate into the circuit or composite gate
        Some Examples are like this:

        X       | circuit
        CX      | circuit([0, 1])
        Measure | CompositeGate

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) Circuit
                2) CompositeGate

        Raise:
            TypeException: the type of other is wrong
        """
        try:
            targets.append(self)
        except Exception:
            raise TypeError("composite gate or circuit", targets)

    def __and__(self, targets):
        """deal the operator '&'

        Use the syntax "gate & int" or "gate & list<int>" to set gate's attribute.
        Special uses when in composite gate's context.

        Some Examples are like this:
        X       & 1
        CX      & [0, 1]

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) int
                2) list<int>

        Raise:
            TypeError: the type of targets is wrong
        """
        _gate = self.copy()

        if isinstance(targets, int):
            assert _gate.is_single()

            _gate.targs = [targets]
        elif isinstance(targets, list):
            assert len(targets) == _gate.controls + _gate.targets

            _gate.cargs = targets[:_gate.controls]
            _gate.targs = targets[_gate.controls:]
        else:
            raise TypeError("int or list<int>", targets)

        if CGATE_LIST:
            CGATE_LIST[-1].append(_gate)
        else:
            return _gate

    def __call__(self):
        """ give parameters for the gate.

        give parameters by "()".
        Some Examples are like this:

        Rz(np.pi / 2)           | qubit
        U3(np.pi / 2, 0, 0)     | qubit

        Args:
            params: give parameters for the gate, it can have following form,
                1) int/float/complex
                2) list<int/float/complex>
                3) tuple<int/float/complex>
        Raise:
            TypeException: the type of params is wrong

        Returns:
            BasicGate: the gate after filled by parameters
        """
        return self.copy()

    def __eq__(self, other):
        if isinstance(other, BasicGate):
            if other.name == self.name:
                return True

        return False

    def update_name(self, qubit_id: str, circuit_idx: int = None):
        qubit_id = qubit_id[:6]
        name_parts = self.name.split('-')
        name_parts[1] = qubit_id

        if circuit_idx is not None:
            name_parts[2] = str(circuit_idx)

        self.name = '-'.join(name_parts)

    def __str__(self):
        """ get gate information """
        gate_info = {
            "name": self.name,
            "controls": self.controls,
            "control_bit": self.cargs,
            "targets": self.targets,
            "target_bit": self.targs,
            "parameters": self.pargs
        }

        return str(gate_info)

    def qasm(self):
        """ generator OpenQASM string for the gate

        Return:
            string: the OpenQASM 2.0 describe of the gate
        """
        if self.type in SPECIAL_GATE_SET[4:]:
            raise KeyError(f"The gate do not support qasm, {self.type}")

        qasm_string = self.qasm_name
        if self.params > 0:
            params = [str(parg) for parg in self.pargs]
            params_string = "(" + ", ".join(params) + ")"

            qasm_string += params_string

        ctargs = [str(ctarg) for ctarg in self.cargs + self.targs]
        ctargs_string = " " + ', '.join(ctargs) + ";\n"
        qasm_string += ctargs_string

        return qasm_string

    def inverse(self):
        """ the inverse of the gate

        Return:
            BasicGate: the inverse of the gate
        """
        return self.copy()

    def commutative(self, goal, eps=1e-7):
        """ decide whether gate is commutative with another gate

        note when the gate is special gates like Unitary, Permutation, Measure and so on, return False.

        Args:
            goal(BasicGate): the target gate
            eps(float): the precision of comparision

        Return:
            bool: True if commutative
        """
        if self.is_special() or goal.is_special():
            return False

        if self.targets > 1 or goal.targets > 1:
            return False

        A = self.target_matrix
        B = goal.target_matrix
        if (
            np.allclose(A, np.identity(2), rtol=eps, atol=eps) or
            np.allclose(B, np.identity(2), rtol=eps, atol=eps)
        ):
            return True

        set_controls = set(self.cargs)
        set_targets = set(self.targs)
        set_goal_controls = set(goal.cargs)
        set_goal_targets = set(goal.targs)

        commutative_set = set_controls.intersection(set_goal_targets)
        if len(commutative_set) > 0 and not goal.is_diagonal():
            return False

        commutative_set = set_goal_controls.intersection(set_targets)
        if len(commutative_set) > 0 and not self.is_diagonal():
            return False

        commutative_set = set_goal_targets.intersection(set_targets)
        if len(commutative_set) > 0 and not np.allclose(A.dot(B), B.dot(A), rtol=1.0e-13, atol=1.0e-13):
            return False

        return True

    def is_single(self) -> bool:
        """ judge whether gate is a one qubit gate(excluding special gate like measure, reset, custom and so on)

        Returns:
            bool: True if it is a one qubit gate
        """
        return self.targets + self.controls == 1

    def is_control_single(self) -> bool:
        """ judge whether gate has one control bit and one target bit

        Returns:
            bool: True if it is has one control bit and one target bit
        """
        return self.controls == 1 and self.targets == 1

    def is_diagonal(self) -> bool:
        """ judge whether gate's matrix is diagonal

        Returns:
            bool: True if gate's matrix is diagonal
        """
        return (
            self.type in DIAGONAL_GATE_SET or
            (self.type == GateType.unitary and self._is_diagonal())
        )

    def _is_diagonal(self) -> bool:
        return np.allclose(np.diag(np.diag(self.matrix)), self.matrix)

    def is_special(self) -> bool:
        """ judge whether gate's is special gate, which is one of
        [Measure, Reset, Barrier, Perm, Unitary, ...]

        Returns:
            bool: True if gate's matrix is special
        """
        return self.type in SPECIAL_GATE_SET

    def copy(self):
        """ return a copy of this gate

        Returns:
            gate(BasicGate): a copy of this gate
        """
        class_name = str(self.__class__.__name__)
        gate = globals()[class_name]()

        if gate.type in SPECIAL_GATE_SET:
            gate.controls = self.controls
            gate.targets = self.targets
            gate.params = self.params

        gate.pargs = copy.deepcopy(self.pargs)
        gate.targs = copy.deepcopy(self.targs)
        gate.cargs = copy.deepcopy(self.cargs)

        if self.assigned_qubits:
            gate.assigned_qubits = copy.deepcopy(self.assigned_qubits)
            gate.update_name(gate.assigned_qubits[0].id)

        return gate

    @staticmethod
    def permit_element(element):
        """ judge whether the type of a parameter is int/float/complex

        for a quantum gate, the parameter should be int/float/complex

        Args:
            element: the element to be judged

        Returns:
            bool: True if the type of element is int/float/complex
        """
        if isinstance(element, int) or isinstance(element, float) or isinstance(element, complex):
            return True
        else:
            tp = type(element)
            if tp == np.int64 or tp == np.float64 or tp == np.complex128:
                return True
            return False


class HGate(BasicGate):
    """ Hadamard gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.h
        )

        self.matrix = np.array([
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1 / np.sqrt(2), -1 / np.sqrt(2)]
        ], dtype=np.complex128)


H = HGate()


class SGate(BasicGate):
    """ S gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.s
        )

        self.matrix = np.array([
            [1, 0],
            [0, 1j]
        ], dtype=np.complex128)


S = SGate()


class SDaggerGate(BasicGate):
    """ The conjugate transpose of Phase gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.sdg
        )

        self.matrix = np.array([
            [1, 0],
            [0, -1j]
        ], dtype=np.complex128)


S_dagger = SDaggerGate()


class XGate(BasicGate):
    """ Pauli-X gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.x
        )

        self.matrix = np.array([
            [0, 1],
            [1, 0]
        ], dtype=np.complex128)


X = XGate()


class YGate(BasicGate):
    """ Pauli-Y gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.y
        )

        self.matrix = np.array([
            [0, -1j],
            [1j, 0]
        ], dtype=np.complex128)


Y = YGate()


class ZGate(BasicGate):
    """ Pauli-Z gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.z
        )

        self.matrix = np.array([
            [1, 0],
            [0, -1]
        ], dtype=np.complex128)


Z = ZGate()


class SXGate(BasicGate):
    """ sqrt(X) gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.sx
        )

        self.matrix = np.array([
            [1 / np.sqrt(2), -1j / np.sqrt(2)],
            [-1j / np.sqrt(2), 1 / np.sqrt(2)]
        ], dtype=np.complex128)

    def inverse(self):
        """ change it be rx gate"""
        _Rx = RxGate([-np.pi / 2])
        _Rx.targs = copy.deepcopy(self.targs)

        return _Rx


SX = SXGate()


class SYGate(BasicGate):
    """ sqrt(Y) gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.sy
        )

        self.matrix = np.array([
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2)]
        ], dtype=np.complex128)

    def inverse(self):
        """ change it to be ry gate"""
        _Ry = RyGate([-np.pi / 2])
        _Ry.targs = copy.deepcopy(self.targs)

        return _Ry


SY = SYGate()


class SWGate(BasicGate):
    """ sqrt(W) gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.sw
        )

        self.matrix = np.array([
            [1 / np.sqrt(2), -np.sqrt(1j / 2)],
            [np.sqrt(-1j / 2), 1 / np.sqrt(2)]
        ], dtype=np.complex128)

    def inverse(self):
        """ change it be U2 gate"""
        _U2 = U2Gate([3 * np.pi / 4, 5 * np.pi / 4])
        _U2.targs = copy.deepcopy(self.targs)

        return _U2


SW = SWGate()


class IDGate(BasicGate):
    """ Identity gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.id
        )

        self.matrix = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.complex128)


ID = IDGate()


class U1Gate(BasicGate):
    """ Diagonal single-qubit gate """
    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=0,
            targets=1,
            params=1,
            type=GateType.u1
        )

        self.pargs = params

    def __call__(self, alpha):
        if not self.permit_element(alpha):
            raise TypeError("int/float/complex", alpha)

        return U1Gate([alpha])

    @property
    def matrix(self):
        return np.array([
            [1, 0],
            [0, np.exp(1j * self.pargs[0])]
        ], dtype=np.complex128)

    def inverse(self):
        _U1 = self.copy()
        _U1.pargs = [-self.pargs[0]]

        return _U1


U1 = U1Gate()


class U2Gate(BasicGate):
    """ One-pulse single-qubit gate """
    def __init__(self, params: list = [np.pi / 2, np.pi / 2]):
        super().__init__(
            controls=0,
            targets=1,
            params=2,
            type=GateType.u2
        )

        self.pargs = params

    def __call__(self, alpha, beta):
        params = [alpha, beta]

        for param in params:
            if not self.permit_element(param):
                raise TypeError("int/float/complex", param)

        return U2Gate(params)

    @property
    def matrix(self):
        sqrt2 = 1 / np.sqrt(2)
        return np.array([
            [1 * sqrt2,
             -np.exp(1j * self.pargs[1]) * sqrt2],
            [np.exp(1j * self.pargs[0]) * sqrt2,
             np.exp(1j * (self.pargs[0] + self.pargs[1])) * sqrt2]
        ], dtype=np.complex128)

    def inverse(self):
        _U2 = self.copy()
        _U2.pargs = [np.pi - self.pargs[1], np.pi - self.pargs[0]]

        return _U2


U2 = U2Gate()


class U3Gate(BasicGate):
    """ Two-pulse single-qubit gate """
    def __init__(self, params: list = [0, 0, np.pi / 2]):
        super().__init__(
            controls=0,
            targets=1,
            params=3,
            type=GateType.u3
        )

        self.pargs = params

    def __call__(self, alpha, beta, gamma):
        params = [alpha, beta, gamma]

        for param in params:
            if not self.permit_element(param):
                raise TypeError("int/float/complex", param)

        return U3Gate(params)

    @property
    def matrix(self):
        return np.array([
            [np.cos(self.pargs[0] / 2),
             -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2)],
            [np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
             np.exp(1j * (self.pargs[1] + self.pargs[2])) * np.cos(self.pargs[0] / 2)]
        ], dtype=np.complex128)

    def inverse(self):
        _U3 = self.copy()
        _U3.pargs = [self.pargs[0], np.pi - self.pargs[2], np.pi - self.pargs[1]]

        return _U3


U3 = U3Gate()


class RxGate(BasicGate):
    """ Rotation around the x-axis gate """
    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=0,
            targets=1,
            params=1,
            type=GateType.rx
        )

        self.pargs = params

    def __call__(self, alpha):
        if not self.permit_element(alpha):
            raise TypeError("int/float/complex", alpha)

        return RxGate([alpha])

    @property
    def matrix(self):
        return np.array([
            [np.cos(self.parg / 2), 1j * -np.sin(self.parg / 2)],
            [1j * -np.sin(self.parg / 2), np.cos(self.parg / 2)]
        ], dtype=np.complex128)

    def inverse(self):
        _Rx = self.copy()
        _Rx.pargs = [-self.pargs[0]]

        return _Rx


Rx = RxGate()


class RyGate(BasicGate):
    """ Rotation around the y-axis gate """
    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=0,
            targets=1,
            params=1,
            type=GateType.ry
        )

        self.pargs = params

    def __call__(self, alpha):
        if not self.permit_element(alpha):
            raise TypeError("int/float/complex", alpha)

        return RyGate([alpha])

    @property
    def matrix(self):
        return np.array([
            [np.cos(self.pargs[0] / 2), -np.sin(self.pargs[0] / 2)],
            [np.sin(self.pargs[0] / 2), np.cos(self.pargs[0] / 2)],
        ], dtype=np.complex128)

    def inverse(self):
        _Ry = self.copy()
        _Ry.pargs = [-self.pargs[0]]

        return _Ry


Ry = RyGate()


class RzGate(BasicGate):
    """ Rotation around the z-axis gate """
    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=0,
            targets=1,
            params=1,
            type=GateType.rz
        )

        self.pargs = params

    def __call__(self, alpha):
        if not self.permit_element(alpha):
            raise TypeError("int/float/complex", alpha)

        return RzGate([alpha])

    @property
    def matrix(self):
        return np.array([
            [np.exp(-self.parg / 2 * 1j), 0],
            [0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    def inverse(self):
        _Rz = self.copy()
        _Rz.pargs = [-self.pargs[0]]

        return _Rz


Rz = RzGate()


class TGate(BasicGate):
    """ T gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.t
        )

        self.matrix = np.array([
            [1, 0],
            [0, 1 / np.sqrt(2) + 1j * 1 / np.sqrt(2)]
        ], dtype=np.complex128)

    def inverse(self):
        """ change it be tdg gate"""
        _Tdagger = TDaggerGate()
        _Tdagger.targs = copy.deepcopy(self.targs)
        _Tdagger.assigned_qubits = copy.deepcopy(self.assigned_qubits)

        return _Tdagger


T = TGate()


class TDaggerGate(BasicGate):
    """ The conjugate transpose of T gate """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.tdg
        )

        self.matrix = np.array([
            [1, 0],
            [0, 1 / np.sqrt(2) + 1j * -1 / np.sqrt(2)]
        ], dtype=np.complex128)

    def inverse(self):
        """ change it to be t gate """
        _Tgate = TGate()
        _Tgate.targs = copy.deepcopy(self.targs)
        _Tgate.assigned_qubits = copy.deepcopy(self.assigned_qubits)

        return _Tgate


T_dagger = TDaggerGate()


class PhaseGate(BasicGate):
    """ Phase gate """
    def __init__(self, params: list = [0]):
        super().__init__(
            controls=0,
            targets=1,
            params=1,
            type=GateType.phase
        )

        self.pargs = params

    def __call__(self, alpha):
        if not self.permit_element(alpha):
            raise TypeError("int/float/complex", alpha)

        return PhaseGate([alpha])

    @property
    def matrix(self):
        return np.array([
            [np.exp(self.parg * 1j), 0],
            [0, np.exp(self.parg * 1j)]
        ], dtype=np.complex128)

    def inverse(self):
        _Phase = self.copy()
        _Phase.pargs = [-self.parg]

        return _Phase


Phase = PhaseGate()


class CZGate(BasicGate):
    """ controlled-Z gate """
    def __init__(self):
        super().__init__(
            controls=1,
            targets=1,
            params=0,
            type=GateType.cz
        )

        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=np.complex128)

        self._target_matrix = np.array([
            [1, 0],
            [0, -1]
        ], dtype=np.complex128)

    @property
    def target_matrix(self):
        return self._target_matrix


CZ = CZGate()


class CXGate(BasicGate):
    """ controlled-X gate """
    def __init__(self):
        super().__init__(
            controls=1,
            targets=1,
            params=0,
            type=GateType.cx
        )

        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)

        self._target_matrix = np.array([
            [0, 1],
            [1, 0]
        ], dtype=np.complex128)

    @property
    def target_matrix(self):
        return self._target_matrix


CX = CXGate()


class CYGate(BasicGate):
    """ controlled-Y gate """
    def __init__(self):
        super().__init__(
            controls=1,
            targets=1,
            params=0,
            type=GateType.cy
        )

        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ], dtype=np.complex128)

        self._target_matrix = np.array([
            [0, -1j],
            [1j, 0]
        ], dtype=np.complex128)

    @property
    def target_matrix(self):
        return self._target_matrix


CY = CYGate()


class CHGate(BasicGate):
    """ controlled-Hadamard gate """
    def __init__(self):
        super().__init__(
            controls=1,
            targets=1,
            params=0,
            type=GateType.ch
        )

        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]
        ], dtype=np.complex128)

        self._target_matrix = np.array([
            [1 / np.sqrt(2), 1 / np.sqrt(2)],
            [1 / np.sqrt(2), -1 / np.sqrt(2)]
        ], dtype=np.complex128)

    @property
    def target_matrix(self):
        return self._target_matrix


CH = CHGate()


class CRzGate(BasicGate):
    """ controlled-Rz gate """

    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=1,
            targets=1,
            params=1,
            type=GateType.crz
        )

        self.pargs = params

    def __call__(self, alpha):
        if not self.permit_element(alpha):
            raise TypeError("int/float/complex", alpha)

        return CRzGate([alpha])

    @property
    def matrix(self):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-self.parg / 2 * 1j), 0],
            [0, 0, 0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    @property
    def target_matrix(self) -> np.ndarray:
        return np.array([
            [np.exp(-self.parg / 2 * 1j), 0],
            [0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    def inverse(self):
        _CRz = self.copy()
        _CRz.pargs = [-self.pargs[0]]

        return _CRz


CRz = CRzGate()


class CU1Gate(BasicGate):
    """ Controlled-U1 gate """
    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=1,
            targets=1,
            params=1,
            type=GateType.cu1
        )

        self.pargs = params

    def __call__(self, alpha):
        if not self.permit_element(alpha):
            raise TypeError("int/float/complex", alpha)

        return CU1Gate([alpha])

    @property
    def matrix(self):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * self.pargs[0])]
        ], dtype=np.complex128)

    @property
    def target_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, np.exp(1j * self.pargs[0])]
        ], dtype=np.complex128)

    def inverse(self):
        _CU1 = self.copy()
        _CU1.pargs = [-self.pargs[0]]

        return _CU1


CU1 = CU1Gate()


class CU3Gate(BasicGate):
    """ Controlled-U3 gate """
    def __init__(self, params: list = [np.pi / 2, 0, 0]):
        super().__init__(
            controls=1,
            targets=1,
            params=3,
            type=GateType.cu3
        )

        self.pargs = params

    def __call__(self, alpha, beta, gamma):
        params = [alpha, beta, gamma]

        for param in params:
            if not self.permit_element(param):
                raise TypeError("int/float/complex", param)

        return CU3Gate(params)

    @property
    def matrix(self):
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.cos(self.pargs[0] / 2), -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2)],
            [0, 0, np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
             np.exp(1j * (self.pargs[1] + self.pargs[2])) * np.cos(self.pargs[0] / 2)]
        ], dtype=np.complex128)

    @property
    def target_matrix(self) -> np.ndarray:
        return np.array([
            [np.cos(self.pargs[0] / 2), -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2)],
            [np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
             np.exp(1j * (self.pargs[1] + self.pargs[2])) * np.cos(self.pargs[0] / 2)]
        ], dtype=np.complex128)

    def inverse(self):
        _CU3 = self.copy()
        _CU3.pargs = [self.pargs[0], np.pi - self.pargs[2], np.pi - self.pargs[1]]

        return _CU3


CU3 = CU3Gate()


class FSimGate(BasicGate):
    """ fSim gate """
    def __init__(self, params: list = [np.pi / 2, 0]):
        super().__init__(
            controls=0,
            targets=2,
            params=2,
            type=GateType.fsim
        )

        self.pargs = params

    def __call__(self, alpha, beta):
        params = [alpha, beta]
        for param in params:
            if not self.permit_element(param):
                raise TypeError("int/float/complex", param)

        return FSimGate(params)

    @property
    def matrix(self):
        costh = np.cos(self.pargs[0])
        sinth = np.sin(self.pargs[0])
        phi = self.pargs[1]

        return np.array([
            [1, 0, 0, 0],
            [0, costh, -1j * sinth, 0],
            [0, -1j * sinth, costh, 0],
            [0, 0, 0, np.exp(-1j * phi)]
        ], dtype=np.complex128)

    def inverse(self):
        _FSim = self.copy()
        _FSim.pargs = [-self.pargs[0], -self.pargs[1]]

        return _FSim


FSim = FSimGate()


class RxxGate(BasicGate):
    """ Rxx gate """
    def __init__(self, params: list = [0]):
        super().__init__(
            controls=0,
            targets=2,
            params=1,
            type=GateType.Rxx
        )

        self.pargs = params

    def __call__(self, alpha):
        if not self.permit_element(alpha):
            raise TypeError("int/float/complex", alpha)

        return RxxGate([alpha])

    @property
    def matrix(self):
        costh = np.cos(self.parg / 2)
        sinth = np.sin(self.parg / 2)

        return np.array([
            [costh, 0, 0, -1j * sinth],
            [0, costh, -1j * sinth, 0],
            [0, -1j * sinth, costh, 0],
            [-1j * sinth, 0, 0, costh]
        ], dtype=np.complex128)

    def inverse(self):
        _Rxx = self.copy()
        _Rxx.pargs = [-self.parg]

        return _Rxx


Rxx = RxxGate()


class RyyGate(BasicGate):
    """ Ryy gate """
    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=0,
            targets=2,
            params=1,
            type=GateType.Ryy
        )

        self.pargs = params

    def __call__(self, alpha):
        if not self.permit_element(alpha):
            raise TypeError("int/float/complex", alpha)

        return RyyGate([alpha])

    @property
    def matrix(self):
        costh = np.cos(self.parg / 2)
        sinth = np.sin(self.parg / 2)

        return np.array([
            [costh, 0, 0, 1j * sinth],
            [0, costh, -1j * sinth, 0],
            [0, -1j * sinth, costh, 0],
            [1j * sinth, 0, 0, costh]
        ], dtype=np.complex128)

    def inverse(self):
        _Ryy = self.copy()
        _Ryy.pargs = [-self.parg]

        return _Ryy


Ryy = RyyGate()


class RzzGate(BasicGate):
    """ Rzz gate """
    def __init__(self, params: list = [np.pi / 2]):
        super().__init__(
            controls=0,
            targets=2,
            params=1,
            type=GateType.Rzz
        )

        self.pargs = params

    def __call__(self, alpha):
        if not self.permit_element(alpha):
            raise TypeError("int/float/complex", alpha)

        return RzzGate([alpha])

    @property
    def matrix(self):
        expth = np.exp(0.5j * self.parg)
        sexpth = np.exp(-0.5j * self.parg)

        return np.array([
            [sexpth, 0, 0, 0],
            [0, expth, 0, 0],
            [0, 0, expth, 0],
            [0, 0, 0, sexpth]
        ], dtype=np.complex128)

    def inverse(self):
        _Rzz = self.copy()
        _Rzz.pargs = [-self.parg]

        return _Rzz


Rzz = RzzGate()


class MeasureGate(BasicGate):
    """ z-axis Measure gate

    Measure one qubit along z-axis.
    After acting on the qubit(circuit flush), the qubit get the value 0 or 1
    and the amplitude changed by the result.
    """

    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.measure
        )

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("try to get the matrix of measure gate")


Measure = MeasureGate()


class ResetGate(BasicGate):
    """ Reset gate

    Reset the qubit into 0 state,
    which change the amplitude
    """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.reset
        )

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("try to get the matrix of reset gate")


Reset = ResetGate()


class BarrierGate(BasicGate):
    """ Barrier gate

    In IBMQ, barrier gate forbid the optimization cross the gate,
    It is invalid in out circuit now.
    """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=1,
            params=0,
            type=GateType.barrier
        )

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("try to get the matrix of barrier gate")


Barrier = BarrierGate()


class SwapGate(BasicGate):
    """ Swap gate

    In the computation, it will not change the amplitude.
    Instead, it change the index of a Tangle.
    """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=2,
            params=0,
            type=GateType.swap
        )

        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)


Swap = SwapGate()


# PermGate class -- no qasm
class PermGate(BasicGate):
    """ Permutation gate

    A special gate defined in our circuit,
    It can change an n-qubit qureg's amplitude by permutaion,
    the parameter is a 2^n list describes the permutation.
    """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=0,
            params=0,
            type=GateType.perm
        )

    def __call__(self, targets: int, params: list):
        """ pass permutation to the gate

        the length of permutaion must be n, and should be a permutation for [0, n) without repeat

        Args:
            params(list): the permutation parameters

        Returns:
            PermGate: the gate after filled by parameters
        """
        if not isinstance(params, list) or not isinstance(targets, int):
            raise TypeError(f"targets must be int not {type(targets)}, and params must be list not {type(params)}")

        assert len(params) == targets, "the length of params must equal to targets"

        _gate = self.copy()
        _gate.targets = targets
        _gate.params = targets
        for idx in params:
            if not isinstance(idx, int) or idx < 0 or idx >= _gate.targets:
                raise Exception("the element in the list should be integer")

            if idx in _gate.pargs:
                raise Exception("the list should be a permutation for [0, n) without repeat")

            _gate.pargs.append(idx)

        return _gate

    def inverse(self):
        _gate = self.copy()
        _gate.pargs = [self.targets - 1 - p for p in self.pargs]

        return _gate


Perm = PermGate()


class ControlPermMulDetailGate(BasicGate):
    """ controlled-Permutation gate

    This gate is used to implement oracle in the order-finding algorithm
    """
    def __init__(self):
        super().__init__(
            controls=1,
            targets=0,
            params=0,
            type=GateType.control_perm_detail
        )

    def __call__(self, a: int, N: int):
        """ pass parameters to the gate

        give parameters a, N to the gate

        Args:
            a (int): the integer between 2 and N - 1
            N (int): a positive integer

        Returns:
            ControlPermMulDetailGate: the gate after filled by parameters
        """
        if not isinstance(a, int) or not isinstance(N, int):
            raise TypeError("a and N must be integer.")

        _gate = self.copy()
        n = int(np.ceil(np.log2(N)))

        _gate.targets = n
        _gate.params = 1 << (n + 1)
        for idx in range(1 << (_gate.targets + _gate.controls)):
            idxx, controlxx = idx // 2, idx % 2
            if controlxx == 0:
                _gate.pargs.append(idx)
            else:
                t_idx = idx if idxx >= N else ((idxx * a % N) << 1) + controlxx
                _gate.pargs.append(t_idx)

        return _gate

    def inverse(self):
        from QuICT.algorithm.quantum_algorithm.shor.utility import ex_gcd

        gcd_arr = [0, 1]
        ex_gcd(self.pargs[0], self.pargs[1], gcd_arr)
        n_inverse = (gcd_arr[0] % self.pargs[1] + self.pargs[1]) % self.pargs[1]

        return self(n_inverse, self.pargs[1])


ControlPermMulDetail = ControlPermMulDetailGate()


class PermShiftGate(BasicGate):
    """ act an increase or subtract operate with modulus.

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self):
        super().__init__(
            controls=0,
            targets=0,
            params=0,
            type=GateType.perm_shift
        )

    def __call__(self, a: int, N: int):
        """ pass parameters to the gate

        give parameters (params, N) to the gate

        Args:
            a(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermShiftGate: the gate after filled by parameters
        """
        if not isinstance(a, int) or not isinstance(N, int):
            raise TypeError(f"params and N must be int. {type(N)}, {type(a)}")

        if N <= 0:
            raise Exception("the modulus should be integer")

        _gate = self.copy()
        n = int(round(np.log2(N)))
        _gate.params = 1 << n
        _gate.targets = n
        for idx in range(1 << _gate.targets):
            idxx, controlxx = idx // 2, idx % 2
            if controlxx == 0:
                _gate.pargs.append(idx)
            else:
                t_idx = idx if idxx < N else ((((idxx + a) % N + N) % N) << 1) + controlxx
                _gate.pargs.append(t_idx)

        return _gate


PermShift = PermShiftGate()


class ControlPermShiftGate(BasicGate):
    """ Controlled-PermShiftGate

    PermShiftGate with a control bit

    """

    def __init__(self):
        super().__init__(
            controls=1,
            targets=0,
            params=0,
            type=GateType.control_perm_shift
        )

    def __call__(self, a: int, N: int):
        """ pass parameters to the gate

        give parameters (params, N) to the gate

        Args:
            params(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermShiftGate: the gate after filled by parameters
        """
        if not isinstance(a, int) or not isinstance(N, int):
            raise TypeError(f"params and N must be int. {type(N)}, {type(a)}")

        if N <= 0:
            raise Exception("the modulus should be integer")

        _gate = self.copy()
        n = int(np.ceil(np.log2(N)))
        _gate.params = 1 << (n + 1)
        _gate.targets = n
        for idx in range(1 << (_gate.targets + _gate.controls)):
            idxx = idx // 2
            controlxx = idx % 2
            if controlxx == 0:
                _gate.pargs.append(idx)
            else:
                t_idx = idx if idxx < N else ((((idxx + a) % N + N) % N) << 1) + controlxx
                _gate.pargs.append(t_idx)

        return _gate


ControlPermShift = ControlPermShiftGate()


class PermMulGate(BasicGate):
    """ act an multiply operate with modulus.

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self):
        super().__init__(
            controls=0,
            targets=0,
            params=0,
            type=GateType.perm_mul
        )

    def __call__(self, a: int, N: int):
        """ pass parameters to the gate

        give parameters (a, N) to the gate

        Args:
            a(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermMulGate: the gate after filled by parameters
        """
        if not isinstance(a, int) or not isinstance(N, int):
            raise TypeError(f"Input must be int. {type(a)}, {type(N)}")

        assert (N > 0 and a > 0), "Input must be positive integer."

        if np.gcd(a, N) != 1:
            raise Exception(f"params and N should be co-prime, but {a} and {N} are not.")

        a = a % N
        n = int(round(np.log2(N)))
        if (1 << n) < N:
            n = n + 1

        _gate = self.copy()
        _gate.params = 1 << n
        _gate.targets = n
        for idx in range(N):
            _gate.pargs.append(idx * a % N)

        for idx in range(N, 1 << n):
            _gate.pargs.append(idx)

        return _gate


PermMul = PermMulGate()


class ControlPermMulGate(BasicGate):
    """ a controlled-PermMul Gate """
    def __init__(self):
        super().__init__(
            controls=1,
            targets=0,
            params=0,
            type=GateType.control_perm_mul
        )

    def __call__(self, a: int, N: int):
        """ pass parameters to the gate

        give parameters (a, N) to the gate

        Args:
            a(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            ControlPermMulGate: the gate after filled by parameters
        """
        if not isinstance(a, int) or not isinstance(N, int):
            raise TypeError(f"Input must be int. {type(a)}, {type(N)}")

        assert (N > 0 and a > 0), "Input must be positive integer."

        if np.gcd(a, N) != 1:
            raise Exception(f"params and N should be co-prime, but {a} and {N} are not.")

        a = a % N
        n = int(np.ceil(np.log2(N)))

        _gate = self.copy()
        _gate.params = 1 << (n + 1)
        _gate.targets = n
        for idx in range(1 << (_gate.targets + _gate.controls)):
            idxx, controlxx = idx // 2, idx % 2
            if controlxx == 0:
                _gate.pargs.append(idx)
            else:
                t_idx = idx if idxx >= N else ((idxx * a % N) << 1) + controlxx
                _gate.pargs.append(t_idx)

        return _gate


ControlPermMul = ControlPermMulGate()


class PermFxGate(BasicGate):
    """ act an Fx oracle on a qureg

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self):
        super().__init__(
            controls=0,
            targets=0,
            params=0,
            type=GateType.perm_fx
        )

    def __call__(self, n: int, params: list):
        """ pass Fx to the gate

        Fx should be a 2^n list that represent a boolean function
        {0, 1}^n -> {0, 1}

        Args:
            n (int): the number of targets
            params (list[int]): the list of index, and the index represent which should be 1.

        Returns:
            PermFxGate: the gate after filled by parameters
        """
        if not isinstance(params, list) or not isinstance(n, int):
            raise TypeError(f"n must be int {type(n)}, params must be list {type(params)}")

        N = 1 << n
        for p in params:
            if p >= N:
                raise Exception("the params should be less than N")

        _gate = self.copy()
        _gate.params = 1 << (n + 1)
        _gate.targets = n + 1
        for idx in range(1 << _gate.targets):
            if idx >> 1 in params:
                _gate.pargs.append(idx ^ 1)
            else:
                _gate.pargs.append(idx)

        return _gate


PermFx = PermFxGate()


class UnitaryGate(BasicGate):
    """ Custom gate

    act an unitary matrix on the qureg,
    the parameters is the matrix

    """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=0,
            params=0,
            type=GateType.unitary
        )

    def __call__(self, params: np.array):
        """ pass the unitary matrix

        Args:
            params(np.array/list): contain 2^n * 2^n elements, which
            form an unitary matrix.

        Returns:
            UnitaryGateGate: the gate after filled by parameters
        """
        _u = UnitaryGate()

        if isinstance(params, list):
            params = np.array(params, dtype=np.complex128)

        matrix_size = params.size
        if matrix_size == 0:
            raise Exception("the list or tuple passed in shouldn't be empty")

        length, width = params.shape
        if length != width:
            N = int(np.log2(matrix_size))
            assert N ^ 2 == matrix_size, "the shape of unitary matrix should be square."

            params = params.reshape(N, N)

        n = int(np.log2(params.shape[0]))
        if (1 << n) != params.shape[0]:
            raise Exception("the length of list should be the square of power(2, n)")

        _u.targets = n
        _u.matrix = params.astype(np.complex128)
        return _u

    def copy(self):
        gate = super().copy()
        gate.matrix = self.matrix

        return gate

    def inverse(self):
        _U = super().copy()
        inverse_matrix = np.array(
            np.mat(self._matrix).reshape(1 << self.targets, 1 << self.targets).H.reshape(1, -1),
            dtype=np.complex128
        )
        _U.matrix = inverse_matrix
        _U.targets = self.targets

        return _U


Unitary = UnitaryGate()


class ShorInitialGate(BasicGate):
    """ a oracle gate to preparation the initial state before IQFT in Shor algorithm

    backends will preparation the initial state by classical operator
    with a fixed measure result of second register.
    """
    def __init__(self):
        super().__init__(
            controls=0,
            targets=0,
            params=0,
            type=GateType.shor_init
        )

    def __call__(self, x: int, N: int, u: int):
        """ pass the parameters

        Args:
            x (int): the base number
            N (int): exponential
            u (int[0/1]): the measure result of the second register

        Returns:
            ShorInitialGate: the gate after filled by parameters
        """
        params = [x, N, u]
        for param in params:
            if not self.permit_element(param):
                raise TypeError("int/float/complex", param)

        n = 2 * int(np.ceil(np.log2(N)))
        self.targets = n
        self.pargs = params

        return self


ShorInitial = ShorInitialGate()


class CCXGate(BasicGate):
    """ Toffoli gate

    When using this gate, it will be showed as a whole gate
    instend of being split into smaller gate

    """
    def __init__(self):
        super().__init__(
            controls=2,
            targets=1,
            params=0,
            type=GateType.ccx
        )

        self.matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0]
        ], dtype=np.complex128)

        self._target_matrix = np.array([
            [0, 1],
            [1, 0]
        ], dtype=np.complex128)

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate(self.controls + self.targets)
        with cgate:
            H & 2
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [0, 2]
            T_dagger & 2
            CX & [0, 2]
            T & 0
            T & 2
            H & 2

        return cgate


CCX = CCXGate()


class CCRzGate(BasicGate):
    """ controlled-Rz gate with two control bits """
    def __init__(self, params: list = [0]):
        super().__init__(
            controls=2,
            targets=1,
            params=1,
            type=GateType.CCRz
        )

        self.pargs = params

    def __call__(self, alpha):
        if not self.permit_element(alpha):
            raise TypeError("int/float/complex", alpha)

        return CCRzGate([alpha])

    @property
    def matrix(self):
        return np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, np.exp(-self.parg / 2 * 1j), 0],
            [0, 0, 0, 0, 0, 0, 0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    @property
    def target_matrix(self) -> np.ndarray:
        return np.array([
            [np.exp(-self.parg / 2 * 1j), 0],
            [0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    def inverse(self):
        _CCRz = self.copy()
        _CCRz.pargs = -self.parg

        return _CCRz

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate(self.controls + self.targets)
        with cgate:
            CRz(self.parg / 2) & [1, 2]
            CX & [0, 1]
            CRz(-self.parg / 2) & [1, 2]
            CX & [0, 1]
            CRz(self.parg / 2) & [0, 2]

        return cgate


CCRz = CCRzGate()


class QFTGate(BasicGate):
    """ QFT gate """
    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            gateSet = self.build_gate()
            self._matrix = gateSet.matrix()
        return self._matrix

    def __init__(self):
        super().__init__(
            controls=0,
            targets=3,
            params=0,
            type=GateType.qft
        )

    def __call__(self, params=None, name=None):
        """ pass the unitary matrix

        Args:
            params(int): point out the number of bits of the gate


        Returns:
            QFTGate: the QFTGate after filled by target number
        """
        self.targets = params
        self.name = name
        return self

    def inverse(self):
        _IQFT = IQFTGate()
        _IQFT.targs = copy.deepcopy(self.targs)
        _IQFT.targets = self.targets
        return _IQFT

    def build_gate(self, targets):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate(targets)
        with cgate:
            for i in range(targets):
                H & i
                for j in range(i + 1, targets):
                    CRz(2 * np.pi / (1 << j - i + 1)) & [j, i]

        return cgate


QFT = QFTGate()


class IQFTGate(BasicGate):
    """ IQFT gate """

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            gateSet = self.build_gate()
            self._matrix = gateSet.matrix()
        return self._matrix

    def __init__(self):
        super().__init__(
            controls=0,
            targets=3,
            params=0,
            type=GateType.iqft
        )

    def __call__(self, params=None, name=None):
        """ pass the unitary matrix

        Args:
            params(int): point out the number of bits of the gate


        Returns:
            IQFTGate: the IQFTGate after filled by target number
        """
        self.targets = params
        self.name = name
        return self

    def inverse(self):
        _QFT = QFTGate()
        _QFT.targs = copy.deepcopy(self.targs)
        _QFT.targets = self.targets
        return _QFT

    def build_gate(self, targets):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate(targets)
        with cgate:
            for i in range(targets - 1, -1, -1):
                for j in range(targets - 1, i, -1):
                    CRz(-2 * np.pi / (1 << j - i + 1)) & [j, i]
                H & i

        return cgate


IQFT = IQFTGate()


class CSwapGate(BasicGate):
    """ Fredkin gate

    When using this gate, it will be showed as a whole gate
    instend of being split into smaller gate
    """
    def __init__(self):
        super().__init__(
            controls=1,
            targets=2,
            params=0,
            type=GateType.cswap
        )

        self.matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.complex128)

        self._target_matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=np.complex128)

    @property
    def target_matrix(self) -> np.ndarray:
        return self._target_matrix

    def build_gate(self):
        from QuICT.core.gate import CompositeGate

        cgate = CompositeGate(self.controls + self.targets)
        with cgate:
            CX & [2, 1]
            H & 2
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [2, 1]
            T_dagger & 1
            CX & [0, 1]
            T & 1
            CX & [0, 2]
            T_dagger & 2
            CX & [0, 2]
            T & 0
            T & 2
            H & 2
            CX & [2, 1]

        return cgate


CSwap = CSwapGate()
