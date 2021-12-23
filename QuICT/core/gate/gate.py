#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/10 9:04
# @Author  : Han Yu
# @File    : _gate.py
import numpy as np
import copy

from QuICT.core import Circuit, Qureg
from QuICT.core.utils import GateType, SPECIAL_GATE_SET, DIAGONAL_GATE_SET

# TODO: gate name refactoring
class BasicGate(object):
    """ the abstract SuperClass of all basic quantum gates

    All basic quantum gates described in the framework have
    some common attributes and some common functions
    which defined in this class

    Attributes:
        name(str): the unique name of the gate
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

        matrix(np.array): the unitary matrix of the quantum gates act on targets
        computer_matrix(np.array): the unitary matrix of the quantum gates act on controls and targets
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
    def type(self):
        return self._type

    @property
    def controls(self) -> int:
        return self._controls

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

    @property
    def type(self):
        return self._type

    # life cycle
    def __init__(self, controls, targets, params, type):
        self._matrix = None

        self._controls = controls
        self._targets = targets
        self._params = params
        self._cargs = []
        self._targs = []
        self._pargs = []

        self._type = type
        self._qasm_name = str(type.name)
        self._name = str(type)

    def __str__(self):
        return self._type.value

    # TODO: refactoring
    def __or__(self, targets):
        """deal the operator '|'

        Use the syntax "gate | circuit" or "gate | qureg" or "gate | qubit"
        to add the gate into the circuit
        When a one qubit gate act on a qureg or a circuit, it means Adding
        the gate on all the qubit of the qureg or circuit
        Some Examples are like this:

        X       | circuit
        CX      | circuit([0, 1])
        Measure | qureg

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) Circuit
                2) qureg
            name(string): the name of the gate
        Raise:
            TypeException: the type of other is wrong
        """
        if isinstance(targets, Circuit):
            qureg = targets.qubits
        elif isinstance(targets, Qureg):
            qureg = targets
        else:
            raise TypeError("qubit or tuple<qubit, qureg> or qureg or list<qubit, qureg> or circuit", targets)

        gate = self.copy()
        circuit = qureg.circuit
        circuit.append(gate, qureg)

        return self

    #TODO: refactoring
    def __and__(self, targets):
        """deal the operator '&'

        Use the syntax "gate & int" or "gate & list<int>" to add the parameter into the circuit
        Some Examples are like this:

        X       & 1
        CX      & [0, 1]
        Measure & 2

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) int
                2) tuple<qubit, qureg>
                3) list<qubit, qureg>
            name(string): the name of the gate
        Raise:
            TypeException: the type of other is wrong
        """
        try:
            if isinstance(targets, int):
                targets = [targets]
            else:
                targets = list(targets)
            self.affectArgs = targets
        except Exception:
            raise TypeError("int or tuple<int> or list<int>", targets)

        return self.copy()

    def __call__(self, params=None, name=None):
        """ give parameters for the gate

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
        self.pargs = []
        if self.params > 0:
            try:
                params = list(params)
                for element in params:
                    if not self.permit_element(element):
                        raise TypeError("int/float/complex or list<int/float/complex> "
                                        "or tuple<int/float/complex>", params)
                    self.pargs.append(element)
            except Exception:
                raise TypeError("int/float/complex or list<int/float/complex> "
                                "or tuple<int/float/complex>", params)

        if name is not None:
            self._name = str(name)

        return self

    def __eq__(self, other):
        if isinstance(other, BasicGate):
            if other.name == self.name:
                return True
        return False

    # get information of gate
    def gate_info(self):
        """ get gate information """
        gate_info = {
            "name": self.name,
            "control_bit": {self.cargs},
            "target_bit": {self.targs},
            "parameters": {self.pargs}
        }

        return gate_info

    def qasm(self):     # TODO: mv it with qasm builder
        """ generator OpenQASM string for the gate

        Return:
            string: the OpenQASM 2.0 describe of the gate
        """
        if self.qasm_name == 'error':
            return 'error'
        qasm_string = self.qasm_name
        if self.params > 0:
            qasm_string += '('
            for i in range(len(self.pargs)):
                if i != 0:
                    qasm_string += ', '
                qasm_string += str(self.pargs[i])
            qasm_string += ')'
        qasm_string += ' '
        first_in = True
        for p in self.cargs:
            if not first_in:
                qasm_string += ', '
            else:
                first_in = False
            qasm_string += f'q[{p}]'
        for p in self.targs:
            if not first_in:
                qasm_string += ', '
            else:
                first_in = False
            qasm_string += f'q[{p}]'
        qasm_string += ';\n'

        return qasm_string

    def inverse(self):
        """ the inverse of the gate

        Return:
            BasicGate: the inverse of the gate
        """
        pass

    def commutative(self, goal, eps=1e-7):  # TODO: refactoring
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

        A = np.array(self.matrix).reshape(2, 2)
        B = np.array(goal.matrix).reshape(2, 2)
        if np.allclose(A, np.identity(2), rtol=eps, atol=eps):
            return True
        if np.allclose(B, np.identity(2), rtol=eps, atol=eps):
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

    # gate information
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

        Raise:
            gate must be basic one qubit gate or two qubit gate

        Returns:
            bool: True if gate's matrix is diagonal
        """
        return self.type in DIAGONAL_GATE_SET

    def is_special(self) -> bool:   # TODO: build special gate set
        """ judge whether gate's is special gate, which is one of
        [Measure, Reset, Barrier, Perm, Unitary]

        Returns:
            bool: True if gate's matrix is special
        """
        return self.type in SPECIAL_GATE_SET

    def copy(self, name=None):
        """ return a copy of this gate

        Args:
            name(string): the name of new gate.
                if name is None and self.name is not None
                new gate's name will be self.name, and
                self.name will be None.

        Returns:
            gate(BasicGate): a copy of this gate
        """
        class_name = str(self.__class__.__name__)
        gate = globals()[class_name]()
        gate.pargs = copy.deepcopy(self.pargs)
        gate.targs = copy.deepcopy(self.targs)
        gate.cargs = copy.deepcopy(self.cargs)
        gate.params = self.params

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
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.h
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
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.s
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
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.sdg
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
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.x
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
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.y
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
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.z
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
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.sx
        )

        self.matrix = np.array([
            [1 / np.sqrt(2), -1j / np.sqrt(2)],
            [-1j / np.sqrt(2), 1 / np.sqrt(2)]
        ], dtype=np.complex128)

    def inverse(self):
        """ change it be rx gate"""
        _Rx = RxGate()
        self.matrix = copy.deepcopy(_Rx.matrix)


SX = SXGate()


class SYGate(BasicGate):
    """ sqrt(Y) gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.sy
        )

        self.matrix = np.array([
            [1 / np.sqrt(2), -1 / np.sqrt(2)],
            [1 / np.sqrt(2), 1 / np.sqrt(2)]
        ], dtype=np.complex128)

    def inverse(self):
        """ change it to be ry gate"""
        _Ry = RyGate()
        self.matrix = copy.deepcopy(_Ry.matrix)


SY = SYGate()


class SWGate(BasicGate):
    """ sqrt(W) gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.sw
        )

        self.matrix = np.array([
            [1 / np.sqrt(2), -np.sqrt(1j / 2)],
            [np.sqrt(-1j / 2), 1 / np.sqrt(2)]
        ], dtype=np.complex128)

    def inverse(self):
        """ change it be U2 gate"""
        _U2 = U2Gate()
        self.matrix = copy.deepcopy(_U2.matrix)


SW = SWGate()


class IDGate(BasicGate):
    """ Identity gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.id
        )

        self.matrix = np.array([
            [1, 0],
            [0, 1]
        ], dtype=np.complex128)


ID = IDGate()


class U1Gate(BasicGate):
    """ Diagonal single-qubit gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 1,
            type = GateType.u1
        )

        self.pargs = [np.pi / 2]
        self.build_matrix()

    def build_matrix(self):
        self.matrix = np.array([
            [1, 0],
            [0, np.exp(1j * self.pargs[0])]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [-self.pargs[0]]
        self.build_matrix()


U1 = U1Gate()


class U2Gate(BasicGate):
    """ One-pulse single-qubit gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 2,
            type = GateType.u2
        )

        self.pargs = [np.pi / 2, np.pi / 2]
        self.build_matrix()

    def build_matrix(self):
        sqrt2 = 1 / np.sqrt(2)
        self.matrix = np.array([
            [1 * sqrt2,
             -np.exp(1j * self.pargs[1]) * sqrt2],
            [np.exp(1j * self.pargs[0]) * sqrt2,
             np.exp(1j * (self.pargs[0] + self.pargs[1])) * sqrt2]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [np.pi - self.pargs[1], np.pi - self.pargs[0]]
        self.build_matrix()


U2 = U2Gate()


class U3Gate(BasicGate):
    """ Two-pulse single-qubit gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 3,
            type = GateType.u3
        )
        self.pargs = [0, 0, np.pi / 2]
        self.build_matrix()

    def build_matrix(self):
        self.matrix = np.array([
            [np.cos(self.pargs[0] / 2),
             -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2)],
            [np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
             np.exp(1j * (self.pargs[1] + self.pargs[2])) * np.cos(self.pargs[0] / 2)]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [self.pargs[0], np.pi - self.pargs[2], np.pi - self.pargs[1]]
        self.build_matrix()


U3 = U3Gate()


class RxGate(BasicGate):
    """ Rotation around the x-axis gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 1,
            type = GateType.rx
        )
        self.pargs = [np.pi / 2]
        self.build_matrix()

    def build_matrix(self):
        self.matrix = np.array([
            [np.cos(self.parg / 2),
             1j * -np.sin(self.parg / 2)],
            [1j * -np.sin(self.parg / 2),
             np.cos(self.parg / 2)]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [-self.pargs[0]]
        self.build_matrix()


Rx = RxGate()


class RyGate(BasicGate):
    """ Rotation around the y-axis gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 1,
            type = GateType.ry
        )
        self.pargs = [np.pi / 2]
        self.build_matrix()

    def build_matrix(self):
        self.matirx = np.array([
            [np.cos(self.pargs[0] / 2), -np.sin(self.pargs[0] / 2)],
            [np.sin(self.pargs[0] / 2), np.cos(self.pargs[0] / 2)],
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [-self.pargs[0]]
        self.build_matrix()


Ry = RyGate()


class RzGate(BasicGate):
    """ Rotation around the z-axis gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 1,
            type = GateType.rz
        )

        self.pargs = [np.pi / 2]
        self.build_matrix()

    def build_matrix(self):
        self.matirx = np.array([
            [np.exp(-self.parg / 2 * 1j), 0],
            [0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [-self.pargs[0]]
        self.build_matrix()


Rz = RzGate()


class TGate(BasicGate):
    """ T gate """
    _matrix = np.array([
        [1, 0],
        [0, 1 / np.sqrt(2) + 1j * 1 / np.sqrt(2)]
    ], dtype=np.complex128)

    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.t
        )
        self.matrix = np.array([
            [1, 0],
            [0, 1 / np.sqrt(2) + 1j * 1 / np.sqrt(2)]
        ], dtype=np.complex128)

    def inverse(self):
        """ change it be tdg gate"""
        _Tdagger = TDaggerGate()
        self.matrix = copy.deepcopy(_Tdagger.matrix)


T = TGate()


class TDaggerGate(BasicGate):
    """ The conjugate transpose of T gate """
    _matrix = np.array([
        [1, 0],
        [0, 1 / np.sqrt(2) + 1j * -1 / np.sqrt(2)]
    ], dtype=np.complex128)

    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.tdg
        )
        self.matrix = np.array([
            [1, 0],
            [0, 1 / np.sqrt(2) + 1j * -1 / np.sqrt(2)]
        ], dtype=np.complex128)

    def inverse(self):
        """ change it to be t gate """
        _Tgate = TGate()
        self.matrix = copy.deepcopy(_Tgate.matrix)


T_dagger = TDaggerGate()


class PhaseGate(BasicGate):
    """ Phase gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 1,
            type = GateType.phase
        )

        self.pargs = [0]
        self.build_matrix()

    def build_matrix(self):
        self.matrix = np.array([
            [np.exp(self.parg * 1j), 0],
            [0, np.exp(self.parg * 1j)]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [-self.parg]
        self.build_matrix()


Phase = PhaseGate()


class CZGate(BasicGate):
    """ controlled-Z gate """
    def __init__(self):
        super().__init__(
            controls = 1,
            targets = 1,
            params = 0,
            type = GateType.cz
        )
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=np.complex128)


CZ = CZGate()


class CXGate(BasicGate):
    """ controlled-X gate """
    def __init__(self):
        super().__init__(
            controls = 1,
            targets = 1,
            params = 0,
            type = GateType.cx
        )
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=np.complex128)


CX = CXGate()


class CYGate(BasicGate):
    """ controlled-Y gate """
    def __init__(self):
        super().__init__(
            controls = 1,
            targets = 1,
            params = 0,
            type = GateType.cy
        )

        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ], dtype=np.complex128)


CY = CYGate()


class CHGate(BasicGate):
    """ controlled-Hadamard gate """
    def __init__(self):
        super().__init__(
            controls = 1,
            targets = 1,
            params = 0,
            type = GateType.ch
        )
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
            [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]
        ], dtype=np.complex128)


CH = CHGate()


class CRzGate(BasicGate):
    """ controlled-Rz gate """

    def __init__(self):
        super().__init__(
            controls = 1,
            targets = 1,
            params = 1,
            type = GateType.crz
        )
        self.pargs = [np.pi / 2]
        self.build_matrix()

    def build_matrix(self):
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-self.parg / 2 * 1j), 0],
            [0, 0, 0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [-self.pargs[0]]
        self.build_matrix()


CRz = CRzGate()


class CU1Gate(BasicGate):
    """ Controlled-U1 gate """
    def __init__(self):
        super().__init__(
            controls = 1,
            targets = 1,
            params = 1,
            type = GateType.cu1
        )

        self.pargs = [np.pi / 2]
        self.build_matrix()

    def build_matrix(self):
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * self.pargs[0])]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [-self.pargs[0]]
        self.build_matrix()


CU1 = CU1Gate()


class CU3Gate(BasicGate):
    """ Controlled-U3 gate """
    def __init__(self):
        super().__init__(
            controls = 1,
            targets = 1,
            params = 3,
            type = GateType.cu3
        )

        self.pargs = [np.pi / 2, 0, 0]
        self.build_matrix()

    def build_matrix(self):
        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.cos(self.pargs[0] / 2), -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2)],
            [0, 0, np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
             np.exp(1j * (self.pargs[1] + self.pargs[2])) * np.cos(self.pargs[0] / 2)]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [self.pargs[0], np.pi - self.pargs[2], np.pi - self.pargs[1]]
        self.build_matrix()


CU3 = CU3Gate()


class FSimGate(BasicGate):
    """ fSim gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 2,
            params = 2,
            type = GateType.fsim
        )

        self.pargs = [np.pi / 2, 0]
        self.build_matrix()

    def build_matrix(self):
        costh = np.cos(self.pargs[0])
        sinth = np.sin(self.pargs[0])
        phi = self.pargs[1]

        self.matrix = np.array([
            [1, 0, 0, 0],
            [0, costh, -1j * sinth, 0],
            [0, -1j * sinth, costh, 0],
            [0, 0, 0, np.exp(-1j * phi)]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [-self.pargs[0], -self.pargs[1]]
        self.build_matrix()


FSim = FSimGate()


class RxxGate(BasicGate):
    """ Rxx gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 2,
            params = 1,
            type = GateType.Rxx
        )

        self.pargs = [0]
        self.build_matrix()

    def build_matrix(self):
        costh = np.cos(self.parg / 2)
        sinth = np.sin(self.parg / 2)

        self.matrix = np.array([
            [costh, 0, 0, -1j * sinth],
            [0, costh, -1j * sinth, 0],
            [0, -1j * sinth, costh, 0],
            [-1j * sinth, 0, 0, costh]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [-self.parg]
        self.build_matrix()


Rxx = RxxGate()


class RyyGate(BasicGate):
    """ Ryy gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 2,
            params = 1,
            type = GateType.Ryy
        )

        self.pargs = [np.pi / 2]
        self.build_matrix()

    def build_matrix(self):
        costh = np.cos(self.parg / 2)
        sinth = np.sin(self.parg / 2)

        self.matrix = np.array([
            [costh, 0, 0, 1j * sinth],
            [0, costh, -1j * sinth, 0],
            [0, -1j * sinth, costh, 0],
            [1j * sinth, 0, 0, costh]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [-self.parg]
        self.build_matrix()


Ryy = RyyGate()


class RzzGate(BasicGate):
    """ Rzz gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 2,
            params = 1,
            type = GateType.Rzz
        )

        self.pargs = [np.pi / 2]
        self.build_matrix()

    def build_matrix(self):
        expth = np.exp(0.5j * self.parg)
        sexpth = np.exp(-0.5j * self.parg)

        self.matrix = np.array([
            [sexpth, 0, 0, 0],
            [0, expth, 0, 0],
            [0, 0, expth, 0],
            [0, 0, 0, sexpth]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = [-self.parg]
        self.build_matrix()


Rzz = RzzGate()


class MeasureGate(BasicGate):
    """ z-axis Measure gate

    Measure one qubit along z-axis.
    After acting on the qubit(circuit flush), the qubit get the value 0 or 1
    and the amplitude changed by the result.
    """

    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.measure
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
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.reset
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
            controls = 0,
            targets = 1,
            params = 0,
            type = GateType.barrier
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
            controls = 0,
            targets = 2,
            params = 0,
            type = GateType.swap
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

    # life cycle
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 0,
            params = 0,
            type = GateType.perm
        )

    def __call__(self, params=None, name=None):
        """ pass permutation to the gate

        the length of permutaion must be 2^n,
        by which we can calculate the number of targets

        Args:
            params(list/tuple): the permutation parameters

        Returns:
            PermGate: the gate after filled by parameters
        """
        self.name = name
        self.pargs = []
        if not isinstance(params, list) or not isinstance(params, tuple):
            raise TypeError("list or tuple", params)
        if isinstance(params, tuple):
            params = list(params)
        length = len(params)
        if length == 0:
            raise Exception("list or tuple shouldn't be empty")
        n = int(round(np.log2(length)))
        if (1 << n) != length:
            raise Exception("the length of list or tuple should be the power of 2")
        self.params = length
        self.targets = n
        for idx in params:
            if not isinstance(idx, int) or idx < 0 or idx >= self.params:
                raise Exception("the element in the list/tuple should be integer")
            if idx in self.pargs:
                raise Exception("the list/tuple should be a permutation for [0, 2^n) without repeat")
            self.pargs.append(idx)
        return self

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.array([], dtype=np.complex128)
        for i in range(self.params):
            for j in range(self.params):
                if self.pargs[i] == j:
                    matrix = np.append(matrix, 1)
                else:
                    matrix = np.append(matrix, 0)
        matrix = matrix.reshape(self.params, self.params)
        return matrix

    def inverse(self):
        matrix = [0] * self.params
        i = 0
        for parg in self.pargs:
            matrix[parg] = i
            i += 1
        self.pargs = matrix
        self.params = self.params


Perm = PermGate()


class ControlPermMulDetailGate(BasicGate):
    """ controlled-Permutation gate

    This gate is used to implement oracle in the order-finding algorithm
    """
    def __init__(self):
        super().__init__(
            controls = 1,
            targets = 0,
            params = 0,
            type = GateType.control_perm_detail
        )

    def __call__(self, params=None, name=None):
        """ pass parameters to the gate

        give parameters (a, N) to the gate

        Args:
            params(list/tuple): the oracle's parameters a and N

        Returns:
            ControlPermMulDetailGate: the gate after filled by parameters
        """
        self.name = name
        self.pargs = []
        if not isinstance(params, list) or not isinstance(params, tuple):
            raise TypeError("list or tuple", params)
        if isinstance(params, tuple):
            params = list(params)
        length = len(params)
        if length != 2:
            raise Exception("the list or tuple passed in should contain two values")
        a = params[0]
        N = params[1]
        n = int(np.ceil(np.log2(N)))
        self.params = 2
        self.targets = n
        self.pargs = [a, N]
        return self

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.array([], dtype=np.complex128)
        a = self.pargs[0]
        N = self.pargs[1]
        pargs = []
        for idx in range(1 << (self.targets + self.controls)):
            idxx = idx // 2
            controlxx = idx % 2
            if controlxx == 0:
                pargs.append(idx)
            else:
                if idxx >= N:
                    pargs.append(idx)
                else:
                    pargs.append(((idxx * a % N) << 1) + controlxx)
        for i in range(1 << (self.targets + self.controls)):
            for j in range(1 << (self.targets + self.controls)):
                if pargs[i] == j:
                    matrix = np.append(matrix, 1)
                else:
                    matrix = np.append(matrix, 0)
        matrix = matrix.reshape(1 << (self.targets + self.controls), 1 << (self.targets + self.controls))
        return matrix

    def inverse(self):
        def EX_GCD(a, b, arr):
            if b == 0:
                arr[0] = 1
                arr[1] = 0
                return a
            g = EX_GCD(b, a % b, arr)
            t = arr[0]
            arr[0] = arr[1]
            arr[1] = t - int(a / b) * arr[1]
            return g

        gcd_arr = [0, 1]
        EX_GCD(self.pargs[0], self.pargs[1], gcd_arr)
        n_inverse = (gcd_arr[0] % self.pargs[1] + self.pargs[1]) % self.pargs[1]

        self.pargs = [n_inverse, self.pargs[1]]


ControlPermMulDetail = ControlPermMulDetailGate()


class PermShiftGate(PermGate):
    """ act an increase or subtract operate with modulus.

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 0,
            params = 0,
            type = GateType.perm_shift
        )

    def __call__(self, params=None, N=None, name=None):
        """ pass parameters to the gate

        give parameters (params, N) to the gate

        Args:
            params(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermShiftGate: the gate after filled by parameters
        """
        self.name = name
        if not isinstance(params, int):
            raise TypeError("int", params)
        if N is None:
            raise Exception("PermShift need two parameters")
        if not isinstance(N, int):
            raise TypeError("int", N)

        if N <= 0:
            raise Exception("the modulus should be integer")
        n = int(round(np.log2(N)))
        self.params = N
        self.targets = n
        self.pargs = []
        for idx in range(1 << self.targets):
            idxx = idx // 2
            controlxx = idx % 2
            if controlxx == 0:
                self.pargs.append(idx)
            else:
                if idxx < N:
                    self.pargs.append(idx)
                else:
                    self.pargs.append(((((idxx + params) % N + N) % N) << 1) + controlxx)
        return self


PermShift = PermShiftGate()


class ControlPermShiftGate(PermGate):
    """ Controlled-PermShiftGate

    PermShiftGate with a control bit

    """

    def __init__(self):
        super().__init__(
            controls = 1,
            targets = 0,
            params = 0,
            type = GateType.control_perm_shift
        )

    def __call__(self, params=None, N=None, name=None):
        """ pass parameters to the gate

        give parameters (params, N) to the gate

        Args:
            params(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermShiftGate: the gate after filled by parameters
        """
        self.name = name
        if not isinstance(params, int):
            raise TypeError("int", params)
        if N is None:
            raise Exception("ControlPermShift need two parameters")
        if not isinstance(N, int):
            raise TypeError("int", N)

        if N <= 0:
            raise Exception("the modulus should be integer")
        n = int(np.ceil(np.log2(N)))
        self.params = N
        self.targets = n + 1
        self.pargs = []
        for idx in range(1 << self.targets):
            idxx = idx // 2
            controlxx = idx % 2
            if controlxx == 0:
                self.pargs.append(idx)
            else:
                if idxx < N:
                    self.pargs.append(idx)
                else:
                    self.pargs.append(((((idxx + params) % N + N) % N) << 1) + controlxx)
        return self


ControlPermShift = ControlPermShiftGate()


class PermMulGate(PermGate):
    """ act an multiply operate with modulus.

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 0,
            params = 0,
            type = GateType.perm_mul
        )

    def __call__(self, params=None, N=None, name=None):
        """ pass parameters to the gate

        give parameters (params, N) to the gate

        Args:
            params(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermMulGate: the gate after filled by parameters
        """
        self.name = name
        if not isinstance(params, int):
            raise TypeError("int", params)
        if N is None:
            raise Exception("PermMul need two parameters")
        if not isinstance(N, int):
            raise TypeError("int", N)
        if N <= 0:
            raise Exception("the modulus should be integer")
        if params <= 0:
            raise Exception("the params should be integer")

        if np.gcd(params, N) != 1:
            raise Exception(f"params and N should be co-prime, but {params} and {N} are not.")

        params = params % N

        n = int(round(np.log2(N)))
        if (1 << n) < N:
            n = n + 1
        self.params = N
        self.targets = n
        self.pargs = []
        for idx in range(N):
            self.pargs.append(idx * params % N)
        for idx in range(N, 1 << n):
            self.pargs.append(idx)
        return self


PermMul = PermMulGate()


class ControlPermMulGate(PermGate):
    """ a controlled-PermMul Gate """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 0,
            params = 0,
            type = GateType.control_perm_mul
        )

    def __call__(self, params=None, N=None, name=None):
        """ pass parameters to the gate

        give parameters (params, N) to the gate

        Args:
            params(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            ControlPermMulGate: the gate after filled by parameters
        """
        self.name = name
        if not isinstance(params, int):
            raise TypeError("int", params)
        if N is None:
            raise Exception("PermMul need two parameters")
        if not isinstance(N, int):
            raise TypeError("int", N)
        if N <= 0:
            raise Exception("the modulus should be integer")
        if params <= 0:
            raise Exception("the params should be integer")

        if np.gcd(params, N) != 1:
            raise Exception(f"params and N should be co-prime, but {params} and {N} are not.")

        params = params % N

        n = int(np.ceil(np.log2(N)))
        self.params = N
        self.targets = n + 1
        self.pargs = []
        for idx in range(1 << self.targets):
            idxx = idx // 2
            controlxx = idx % 2
            if controlxx == 0:
                self.pargs.append(idx)
            else:
                if idxx >= N:
                    self.pargs.append(idx)
                else:
                    self.pargs.append(((idxx * params % N) << 1) + controlxx)
        return self


ControlPermMul = ControlPermMulGate()


class PermFxGate(PermGate):
    """ act an Fx oracle on a qureg

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 0,
            params = 0,
            type = GateType.perm_fx
        )

    def __call__(self, params=None, name=None):
        """ pass Fx to the gate

        Fx should be a 2^n list that represent a boolean function
        {0, 1}^n -> {0, 1}

        Args:
            params(list):contain 2^n values which are 0 or 1

        Returns:
            PermFxGate: the gate after filled by parameters
        """
        self.name = name
        if not isinstance(params, list):
            raise TypeError("list", params)
        n = int(round(np.log2(len(params))))
        if len(params) != 1 << n:
            raise Exception("the length of params should be the power of 2")
        N = 1 << n
        for i in range(N):
            if params[i] != 0 and params[i] != 1:
                raise Exception("the range of params should be {0, 1}")

        self.params = 1 << (n + 1)
        self.targets = n + 1
        self.pargs = []

        N_2 = N << 1
        for idx in range(N_2):
            if params[idx >> 1] == 1:
                self.pargs.append(idx ^ 1)
            else:
                self.pargs.append(idx)
        return self


PermFx = PermFxGate()


class UnitaryGate(BasicGate):
    """ Custom gate

    act an unitary matrix on the qureg,
    the parameters is the matrix

    """
    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 0,
            params = 0,
            type = GateType.unitary
        )

    def __call__(self, params=None, name=None):
        """ pass the unitary matrix

        Args:
            params(np.array/list/tuple): contain 2^n * 2^n elements, which
            form an unitary matrix.


        Returns:
            UnitaryGateGate: the gate after filled by parameters
        """
        self.name = name
        if isinstance(params, np.ndarray):
            shape = params.shape
            n2 = shape[0]
            if shape[0] != shape[1]:
                raise Exception("the length of list or tuple should be the square of power(2, n)")
            n = int(round(np.log2(n2)))
            if (1 << n) != n2:
                raise Exception("the length of list or tuple should be the square of power(2, n)")
            self.targets = n
            self.matrix = np.array(params, dtype=np.complex128)
            return self

        if not isinstance(params, list) and not isinstance(params, tuple):
            raise TypeError("list or tuple", params)
        if isinstance(params, tuple):
            params = list(params)
        length = len(params)
        if length == 0:
            raise Exception("the list or tuple passed in shouldn't be empty")
        n2 = int(round(np.sqrt(length)))
        if n2 * n2 != length:
            raise Exception("the length of list or tuple should be the square of power(2, n)")
        n = int(round(np.log2(n2)))
        if (1 << n) != n2:
            raise Exception("the length of list or tuple should be the square of power(2, n)")
        self.targets = n
        self.matrix = np.array(params, dtype=np.complex128).reshape(n2, n2)
        return self

    def copy(self, name=None):
        gate = super().copy(name)
        gate.matrix = self.matrix
        return gate

    def inverse(self):
        inverse_matrix = np.array(
            np.mat(self._matrix).reshape(1 << self.targets, 1 << self.targets).H.reshape(1, -1),
            dtype=np.complex128
        )

        self.matrix = inverse_matrix


Unitary = UnitaryGate()


class ShorInitialGate(BasicGate):
    """ a oracle gate to preparation the initial state before IQFT in Shor algorithm

    backends will preparation the initial state by classical operator
    with a fixed measure result of second register.

    """

    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 0,
            params = 0,
            type = GateType.shor_init
        )

    def __call__(self, params=None, name=None):
        """ pass the parameters

        Args:
            params(list/tuple): contain the parameters x, N and u which indicate
            the base number, exponential and the measure result of the second register.

        Returns:
            ShorInitialGate: the gate after filled by parameters

        """
        self.name = name
        if not isinstance(params, list) and not isinstance(params, tuple):
            raise TypeError("list or tuple", params)
        if isinstance(params, tuple):
            params = list(params)
        length = len(params)
        if length != 3:
            raise Exception("list or tuple passed in should contain three values")
        x = params[0]
        N = params[1]
        u = params[2]
        n = 2 * int(np.ceil(np.log2(N)))
        self.targets = n
        self.pargs = [x, N, u]
        return self


ShorInitial = ShorInitialGate()


# TODO: refactoring complex gate
class ComplexGate(BasicGate):
    """ the abstract SuperClass of all complex quantum gates

    These quantum gates are generally too complex to act on reality quantum
    hardware directly. The class is devoted to give some reasonable synthetize
    of the gates so that user can use these gates as basic gates but get a
    series one-qubit and two-qubit gates in final.

    All complex quantum gates described in the framework have
    some common attributes and some common functions
    which defined in this class.

    Note that the ComplexGate extends the BasicGate

    Note that all subClass must overload the function "build_gate"
    """

    def build_gate(self):
        """ generate BasicGate, affectArgs

        Returns:
            CompositeGate: synthetize result
        """
        pass


class CCXGate(ComplexGate):
    """ Toffoli gate

    When using this gate, it will be showed as a whole gate
    instend of being split into smaller gate

    """
    def __init__(self):
        super().__init__(
            controls = 2,
            targets = 1,
            params = 0,
            type = GateType.ccx
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

    def build_gate(self):
        from .composite_gate import CompositeGate
        qureg = self.affectArgs
        gates = CompositeGate()

        with gates:
            H & qureg[2]
            CX & (qureg[2], qureg[1])
            T_dagger & qureg[1]
            CX & (qureg[0], qureg[1])
            T & qureg[1]
            CX & (qureg[2], qureg[1])
            T_dagger & qureg[1]
            CX & (qureg[0], qureg[1])
            T & qureg[1]
            CX & (qureg[0], qureg[2])
            T_dagger & qureg[2]
            CX & (qureg[0], qureg[2])
            T & qureg[0]
            T & qureg[2]
            H & qureg[2]
        return gates


CCX = CCXGate()


class CCRzGate(ComplexGate):
    """ controlled-Rz gate with two control bits """
    def __init__(self):
        super().__init__(
            controls = 2,
            targets = 1,
            params = 1,
            type = GateType.CCRz
        )

        self.pargs = [0]
        self.build_matrix()

    def build_matrix(self):
        self.matrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, np.exp(-self.parg / 2 * 1j), 0],
            [0, 0, 0, 0, 0, 0, 0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    def inverse(self):
        self.pargs = -self.parg
        self.build_matrix()

    def build_gate(self):
        from .composite_gate import CompositeGate
        qureg = self.affectArgs
        gates = CompositeGate()

        with gates:
            CRz(self.parg / 2) & (qureg[1], qureg[2])
            CX & (qureg[0], qureg[1])
            CRz(-self.parg / 2) & (qureg[1], qureg[2])
            CX & (qureg[0], qureg[1])
            CRz(self.parg / 2) & (qureg[0], qureg[2])
        return gates


CCRz = CCRzGate()


class QFTGate(ComplexGate):
    """ QFT gate """
    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            gateSet = self.build_gate()
            self._matrix = gateSet.matrix()
        return self._matrix

    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 3,
            params = 0,
            type = GateType.qft
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
        from .composite_gate import CompositeGate
        self.targets = targets
        qureg = [i for i in range(targets)]
        gates = CompositeGate()

        with gates:
            for i in range(self.targets):
                H & qureg[i]
                for j in range(i + 1, self.targets):
                    CRz(2 * np.pi / (1 << j - i + 1)) & (qureg[j], qureg[i])
        return gates


QFT = QFTGate()


class IQFTGate(ComplexGate):
    """ IQFT gate """

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            gateSet = self.build_gate()
            self._matrix = gateSet.matrix()
        return self._matrix

    def __init__(self):
        super().__init__(
            controls = 0,
            targets = 3,
            params = 0,
            type = GateType.iqft
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
        from .composite_gate import CompositeGate
        self.targets = targets
        qureg = [i for i in range(targets)]
        gates = CompositeGate()

        with gates:
            for i in range(self.targets - 1, -1, -1):
                for j in range(self.targets - 1, i, -1):
                    CRz(-2 * np.pi / (1 << j - i + 1)) & (qureg[j], qureg[i])
                H & qureg[i]
        return gates


IQFT = IQFTGate()


class CSwapGate(ComplexGate):
    """ Fredkin gate

    When using this gate, it will be showed as a whole gate
    instend of being split into smaller gate
    """
    def __init__(self):
        super().__init__(
            controls = 1,
            targets = 2,
            params = 0,
            type = GateType.cswap
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

    def build_gate(self):
        from .composite_gate import CompositeGate
        qureg = self.affectArgs
        gates = CompositeGate()

        with gates:
            CX & (qureg[2], qureg[1])
            H & qureg[2]
            CX & (qureg[2], qureg[1])
            T_dagger & qureg[1]
            CX & (qureg[0], qureg[1])
            T & qureg[1]
            CX & (qureg[2], qureg[1])
            T_dagger & qureg[1]
            CX & (qureg[0], qureg[1])
            T & qureg[1]
            CX & (qureg[0], qureg[2])
            T_dagger & qureg[2]
            CX & (qureg[0], qureg[2])
            T & qureg[0]
            T & qureg[2]
            H & qureg[2]
            CX & (qureg[2], qureg[1])
        return gates


CSwap = CSwapGate()
