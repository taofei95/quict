#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/10 9:04
# @Author  : Han Yu
# @File    : _gate.py

import copy
import string
import functools

import numpy as np

from QuICT.core.exception import TypeException, NotImplementedGateException
# from QuICT.core.gate.gate_builder import GateBuilder
# from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.core.qubit import Qubit, Qureg
from QuICT.core.circuit import Circuit
from .exec_operator import *

# global gate order count
gate_order = 0


def _add_alias(alias, standard_name):
    if alias is not None:
        global GATE_ID
        if isinstance(alias, str):
            GATE_ID[alias] = GATE_ID[standard_name]
        else:
            for nm in alias:
                if nm in GATE_ID:
                    continue
                GATE_ID[nm] = GATE_ID[standard_name]


GATE_STANDARD_NAME_OF = {-1: "Error"}
# Get standard gate name by gate id.


GATE_ID = {"Error": -1}
# Get gate id by gate name. You may use any one of the aliases of this gate.

GATE_ID_CNT = 0
# Gate number counter.

GATE_NAME_TO_INSTANCE = {}
# Gate name mapping

GATE_SET_LIST = []


# add gate into list environment

def gate_implementation(cls):
    global GATE_STANDARD_NAME_OF
    global GATE_ID
    global GATE_ID_CNT

    GATE_STANDARD_NAME_OF[GATE_ID_CNT] = cls
    GATE_ID[cls.__name__] = GATE_ID_CNT
    GATE_ID_CNT += 1

    @functools.wraps(cls)
    def gate_variation(*args, **kwargs):
        return cls(*args, **kwargs)

    return gate_variation


def gate_build_name(gate, name=None):
    global gate_order
    gate_order += 1
    if name is not None:
        if name in GATE_NAME_TO_INSTANCE:
            raise Exception("the gate's name exists.")
        gate.name = name
    else:
        random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        name = f"{gate.__class__.__name__}_{gate_order}_{random_str}"
        while name in GATE_NAME_TO_INSTANCE:
            random_str = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
            name = f"{gate.__class__.__name__}_{gate_order}_{random_str}"
        gate.name = name
    GATE_NAME_TO_INSTANCE[gate.name] = gate


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
    # Attribute
    _matrix = None

    @property
    def name(self) -> str:
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        raise Exception("the matrix shouldn't be changed.")

    @property
    def compute_matrix(self) -> np.ndarray:
        return self.matrix

    @compute_matrix.setter
    def compute_matrix(self, matrix):
        raise Exception("the matrix shouldn't be changed.")

    @property
    def controls(self) -> int:
        return self.__controls

    @controls.setter
    def controls(self, controls: int):
        self.__controls = controls

    @property
    def cargs(self):
        return self.__cargs

    @cargs.setter
    def cargs(self, cargs: list):
        if isinstance(cargs, list):
            self.__cargs = cargs
        else:
            self.__cargs = [cargs]

    @property
    def targets(self) -> int:
        return self.__targets

    @targets.setter
    def targets(self, targets: int):
        self.__targets = targets

    @property
    def targs(self):
        return self.__targs

    @targs.setter
    def targs(self, targs: list):
        if isinstance(targs, list):
            self.__targs = targs
        else:
            self.__targs = [targs]

    @property
    def params(self) -> int:
        return self.__params

    @params.setter
    def params(self, params: int):
        self.__params = params

    @property
    def pargs(self):
        return self.__pargs

    @pargs.setter
    def pargs(self, pargs: list):
        if isinstance(pargs, list):
            self.__pargs = pargs
        else:
            self.__pargs = [pargs]

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
        return self.__qasm_name

    @qasm_name.setter
    def qasm_name(self, qasm_name):
        self.__qasm_name = qasm_name

    @property
    def affectArgs(self):
        args = []
        if self.controls > 0:
            for carg in self.cargs:
                args.append(carg)
        if self.targets > 0:
            for targ in self.targs:
                args.append(targ)
        return args

    @affectArgs.setter
    def affectArgs(self, affectArgs):
        if len(affectArgs) != self.controls + self.targets:
            print(self.__class__.name)
            print(self.controls, self.targets)
            print(affectArgs)
            raise Exception(f"length of affectArgs should equal to {self.controls + self.targets}")
        if self.controls > 0:
            self.cargs = affectArgs[:self.controls]
        self.targs = affectArgs[self.controls:]

    @classmethod
    def type(cls):
        if cls.__name__ in GATE_ID:
            return GATE_ID[cls.__name__]
        else:
            raise NotImplementedGateException(cls.__name__)

    def __init_subclass__(cls, **kwargs):
        return gate_implementation(cls)

    # life cycle
    def __init__(self, alias=None):
        self.__matrix = []
        self.__cargs = []
        self.__targs = []
        self.__pargs = []
        self.__controls = 0
        self.__targets = 0
        self.__params = 0
        self.__qasm_name = 'error'
        self.__name = None
        self.__temp_name = None
        _add_alias(alias=alias, standard_name=self.__class__.__name__)

    # gate behavior
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
                2) qubit
                3) qureg
                4) tuple<qubit, qureg>
                5) list<qubit, qureg>
            name(string): the name of the gate
        Raise:
            TypeException: the type of other is wrong
        """
        try:
            qureg = Qureg(targets)
            circuit = qureg.circuit
            if self.is_single():
                for qubit in qureg:
                    gate = self.copy()
                    circuit.append(gate, qubit)
            else:
                gate = self.copy()
                circuit.append(gate, qureg)
        except Exception:
            raise TypeException("qubit or tuple<qubit, qureg> or qureg or list<qubit, qureg> or circuit", targets)
        return self

    # gate behavior
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
            raise TypeException("int or tuple<int> or list<int>", targets)
        if len(GATE_SET_LIST):
            GATE_SET_LIST[-1].append(self.copy())
        return self

    def __call__(self, params = None, name = None):
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
                params = [params] if self.permit_element(params) else list(params)
                for element in params:
                    if not self.permit_element(element):
                        raise TypeException("int/float/complex or list<int/float/complex> "
                                            "or tuple<int/float/complex>", params)
                    self.pargs.append(element)
            except Exception:
                raise TypeException("int/float/complex or list<int/float/complex> "
                                    "or tuple<int/float/complex>", params)
        if name is not None:
            self.__temp_name = str(name)
        return self

    def __eq__(self, other):
        if isinstance(other, BasicGate):
            if other.name == self.name:
                return True
        return False

    # get information of gate
    def print_info(self):
        """ print the information of the gate

        print the gate's information, including controls, targets and parameters

        """
        if self.name is not None:
            information = self.name
        else:
            information = self.__str__()
        if self.controls != 0:
            information = information + f" control bit:{self.cargs} "
        if self.targets != 0:
            information = information + f" target bit:{self.targs} "
        if self.params != 0:
            information = information + f" parameters:{self.pargs} "
        print(information)

    def qasm(self):
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
        raise Exception("undefined inverse")

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
        if not self.is_single() and not self.is_control_single() and not self.type() == GATE_ID['Swap']:
            raise Exception("only consider one qubit and two qubits")
        if self.is_single():
            matrix = self.matrix
            if abs(matrix[0, 1]) < 1e-10 and abs(matrix[1, 0]) < 1e-10:
                return True
            else:
                return False
        elif self.is_control_single():
            matrix = self.matrix
            if abs(matrix[0, 1]) < 1e-10 and abs(matrix[1, 0]) < 1e-10:
                return True
            else:
                return False
        else:
            return False

    def is_special(self) -> bool:
        """ judge whether gate's is special gate

        Returns:
            bool: True if gate's matrix is special
        """
        if self.type() == GATE_ID["Unitary"]:
            return True
        if self.type() == GATE_ID["Perm"]:
            return True
        if self.type() == GATE_ID["Measure"]:
            return True
        if self.type() == GATE_ID["Barrier"]:
            return True
        return False

    def exec(self, circuit):
        """ execute on the circuit

        should be overloaded by subClass

        Args:
            circuit(Circuit): the circuit this gate act on

        """
        raise Exception("cannot execute: undefined gate")

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
        gate.targets = self.targets
        gate.controls = self.controls
        gate.params = self.params
        if name is not None:
            gate_build_name(gate, name)
        else:
            gate_build_name(gate, self.__temp_name)
            self.__temp_name = None
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
    """ Hadamard gate


    """
    _matrix = np.array([
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)

        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "h"

    def __str__(self):
        return "H gate"

    def inverse(self):
        _H = HGate(alias=None)
        _H.targs = copy.deepcopy(self.targs)
        return _H

    def exec(self, circuit):
        exec_single(self, circuit)


H = HGate(["H"])


class SGate(BasicGate):
    """ Phase gate


    """

    _matrix = np.array([
        [1, 0],
        [0, 1j]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "s"

    def __str__(self):
        return "Phase gate"

    def inverse(self):
        _S_dagger = SDaggerGate(alias=None)
        _S_dagger.targs = copy.deepcopy(self.targs)
        return _S_dagger

    def exec(self, circuit):
        exec_single(self, circuit)


S = SGate(["S"])


class SDaggerGate(BasicGate):
    """ The conjugate transpose of Phase gate


    """
    _matrix = np.array([
        [1, 0],
        [0, -1j]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "sdg"

    def __str__(self):
        return "The conjugate transpose of Phase gate"

    def inverse(self):
        _SBACK = SGate(alias=None)
        _SBACK.targs = copy.deepcopy(self.targs)
        return _SBACK

    def exec(self, circuit):
        exec_single(self, circuit)


S_dagger = SDaggerGate(["S_dagger"])


class XGate(BasicGate):
    """ Pauli-X gate

    """

    _matrix = np.array([
        [0, 1],
        [1, 0]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "x"

    def __str__(self):
        return "Pauli-X gate"

    def inverse(self):
        _X = XGate(alias=None)
        _X.targs = copy.deepcopy(self.targs)
        return _X

    def exec(self, circuit):
        exec_single(self, circuit)


X = XGate(["X"])


class YGate(BasicGate):
    """ Pauli-Y gate

    """

    _matrix = np.array([
        [0, -1j],
        [1j, 0]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "y"

    def __str__(self):
        return "Pauli-Y gate"

    def inverse(self):
        _Y = YGate(alias=None)
        _Y.targs = copy.deepcopy(self.targs)
        return _Y

    def exec(self, circuit):
        exec_single(self, circuit)


Y = YGate(["Y"])


class ZGate(BasicGate):
    """ Pauli-Z gate

    """

    _matrix = np.array([
        [1, 0],
        [0, -1]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "z"

    def __str__(self):
        return "Pauli-Z gate"

    def inverse(self):
        _Z = ZGate(alias=None)
        _Z.targs = copy.deepcopy(self.targs)
        return _Z

    def exec(self, circuit):
        exec_single(self, circuit)


Z = ZGate(alias=["Z"])

class SXGate(BasicGate):
    """ sqrt(X) gate

    """

    _matrix = np.array([
            [1 / np.sqrt(2), -1j / np.sqrt(2)],
            [-1j / np.sqrt(2), 1 / np.sqrt(2)]
        ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "sx"

    def __str__(self):
        return "sqrt(X) gate"

    def inverse(self):
        _Rx = RxGate(alias=None)
        _Rx.targs = copy.deepcopy(self.targs)
        _Rx.pargs = [-np.pi / 2]
        return _Rx

    def exec(self, circuit):
        exec_single(self, circuit)

SX = SXGate(["SX", "sx", "Sx"])

class SYGate(BasicGate):
    """ sqrt(Y) gate

    """

    _matrix = np.array([
        [1 / np.sqrt(2), -1 / np.sqrt(2)],
        [1 / np.sqrt(2), 1 / np.sqrt(2)]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "sy"

    def __str__(self):
        return "sqrt(Y) gate"

    def inverse(self):
        _Ry = RyGate(alias=None)
        _Ry.targs = copy.deepcopy(self.targs)
        _Ry.pargs = [-np.pi / 2]
        return _Ry

    def exec(self, circuit):
        exec_single(self, circuit)

SY = SYGate(["SY", "sy", "Sy"])

class SWGate(BasicGate):
    """ sqrt(W) gate

    """

    _matrix = np.array([
        [1 / np.sqrt(2), -np.sqrt(1j / 2)],
        [np.sqrt(-1j / 2), 1 / np.sqrt(2)]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "sw"

    def __str__(self):
        return "sqrt(W) gate"

    def inverse(self):
        _U2 = U2Gate(alias=None)
        _U2.targs = copy.deepcopy(self.targs)
        _U2.pargs = [3 * np.pi / 4, 5 * np.pi / 4]
        return _U2

    def exec(self, circuit):
        exec_single(self, circuit)

SW = SWGate(["SW", "sw", "Sw"])

class IDGate(BasicGate):
    """ Identity gate

    """

    _matrix = np.array([
        [1, 0],
        [0, 1]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "id"

    def __str__(self):
        return "Identity gate"

    def inverse(self):
        _ID = IDGate(alias=None)
        _ID.targs = copy.deepcopy(self.targs)
        return _ID

    def exec(self, circuit):
        exec_single(self, circuit)


ID = IDGate(["ID"])

class U1Gate(BasicGate):
    """ Diagonal single-qubit gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "u1"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, np.exp(1j * self.pargs[0])]
        ], dtype=np.complex128)

    def __str__(self):
        return "U1 gate"

    def inverse(self):
        _U1 = U1Gate(alias=None)
        _U1.targs = copy.deepcopy(self.targs)
        _U1.pargs = [-self.pargs[0]]
        return _U1

    def exec(self, circuit):
        exec_single(self, circuit)


U1 = U1Gate(["U1"])


class U2Gate(BasicGate):
    """ One-pulse single-qubit gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 2
        self.pargs = [np.pi / 2, np.pi / 2]
        self.qasm_name = "u2"

    @property
    def matrix(self) -> np.ndarray:
        sqrt2 = 1 / np.sqrt(2)
        return np.array([
            [1 * sqrt2,
            -np.exp(1j * self.pargs[1]) * sqrt2],
            [np.exp(1j * self.pargs[0]) * sqrt2,
            np.exp(1j * (self.pargs[0] + self.pargs[1])) * sqrt2]
        ], dtype=np.complex128)

    def __str__(self):
        return "U2 gate"

    def inverse(self):
        _U2 = U2Gate(alias=None)
        _U2.targs = copy.deepcopy(self.targs)
        _U2.pargs = [np.pi - self.pargs[1], np.pi - self.pargs[0]]
        return _U2

    def exec(self, circuit):
        exec_single(self, circuit)


U2 = U2Gate(["U2"])


class U3Gate(BasicGate):
    """ Two-pulse single-qubit gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 3
        self.pargs = [0, 0, np.pi / 2]
        self.qasm_name = "u3"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [np.cos(self.pargs[0] / 2),
            -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2)],
            [np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
            np.exp(1j * (self.pargs[1] + self.pargs[2])) * np.cos(self.pargs[0] / 2)]
        ], dtype=np.complex128)

    def __str__(self):
        return "U3 gate"

    def inverse(self):
        _U3 = U3Gate(alias=None)
        _U3.targs = copy.deepcopy(self.targs)
        _U3.pargs = [self.pargs[0], np.pi - self.pargs[2], np.pi - self.pargs[1]]
        return _U3

    def exec(self, circuit):
        exec_single(self, circuit)


U3 = U3Gate(["U3"])


class RxGate(BasicGate):
    """ Rotation around the x-axis gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "rx"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [np.cos(self.parg / 2),
            1j * -np.sin(self.parg / 2)],
            [1j * -np.sin(self.parg / 2),
            np.cos(self.parg / 2)]
        ], dtype=np.complex128)

    def __str__(self):
        return "Rx gate"

    def inverse(self):
        _Rx = RxGate(alias=None)
        _Rx.targs = copy.deepcopy(self.targs)
        _Rx.pargs = [-self.pargs[0]]
        return _Rx

    def exec(self, circuit):
        exec_single(self, circuit)


Rx = RxGate(["Rx", "RX"])


class RyGate(BasicGate):
    """ Rotation around the y-axis gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "ry"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [np.cos(self.pargs[0] / 2), -np.sin(self.pargs[0] / 2)],
            [np.sin(self.pargs[0] / 2), np.cos(self.pargs[0] / 2)],
        ], dtype=np.complex128)

    def __str__(self):
        return "Ry gate"

    def inverse(self):
        _Ry = RyGate(alias=None)
        _Ry.targs = copy.deepcopy(self.targs)
        _Ry.pargs = [-self.pargs[0]]
        return _Ry

    def exec(self, circuit):
        exec_single(self, circuit)


Ry = RyGate(["Ry", "RY"])


class RzGate(BasicGate):
    """ Rotation around the z-axis gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "rz"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [np.exp(-self.parg / 2 * 1j), 0],
            [0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    def __str__(self):
        return "Rz gate"

    def inverse(self):
        _Rz = RzGate(alias=None)
        _Rz.targs = copy.deepcopy(self.targs)
        _Rz.pargs = [-self.pargs[0]]
        return _Rz

    def exec(self, circuit):
        exec_single(self, circuit)


Rz = RzGate(["Rz", "RZ"])


class TGate(BasicGate):
    """ T gate

    """

    _matrix = np.array([
        [1, 0],
        [0, 1 / np.sqrt(2) + 1j * 1 / np.sqrt(2)]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "t"

    def __str__(self):
        return "T gate"

    def inverse(self):
        _Tdagger = TDaggerGate()
        _Tdagger.targs = copy.deepcopy(self.targs)
        return _Tdagger

    def exec(self, circuit):
        exec_single(self, circuit)


T = TGate(["T"])


class TDaggerGate(BasicGate):
    """ The conjugate transpose of T gate

    """

    _matrix = np.array([
        [1, 0],
        [0, 1 / np.sqrt(2) + 1j * -1 / np.sqrt(2)]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "tdg"

    def __str__(self):
        return "The conjugate transpose of T gate"

    def inverse(self):
        _Tgate = TGate(alias=None)
        _Tgate.targs = copy.deepcopy(self.targs)
        return _Tgate

    def exec(self, circuit):
        exec_single(self, circuit)


T_dagger = TDaggerGate(["T_dagger"])


class PhaseGate(BasicGate):
    """ Phase gate

    """

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [np.exp(self.parg * 1j), 0],
            [0, np.exp(self.parg * 1j)]
        ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [0]
        self.qasm_name = "phase"

    def __str__(self):
        return "Phase gate"

    def inverse(self):
        _Phase = PhaseGate()
        _Phase.targs = copy.deepcopy(self.targs)
        _Phase.pargs = [-self.parg]
        return _Phase

    def exec(self, circuit):
        exec_single(self, circuit)


Phase = PhaseGate(["Phase"])


class CZGate(BasicGate):
    """ controlled-Z gate

    """

    _matrix = np.array([
        [1, 0],
        [0, -1]
    ], dtype=np.complex128)

    _compute_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=np.complex128)

    @property
    def compute_matrix(self) -> np.ndarray:
        return self._compute_matrix

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "cz"

    def __str__(self):
        return "controlled-Z gate"

    def inverse(self):
        _CZ = CZGate(alias=None)
        _CZ.cargs = copy.deepcopy(self.cargs)
        _CZ.targs = copy.deepcopy(self.targs)
        return _CZ

    def exec(self, circuit):
        exec_controlSingle(self, circuit)


CZ = CZGate(["CZ", "Cz"])


class CXGate(BasicGate):
    """ "controlled-X gate"


    """

    _matrix = np.array([
        [0, 1],
        [1, 0]
    ], dtype=np.complex128)

    _compute_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128)

    @property
    def compute_matrix(self) -> np.ndarray:
        return self._compute_matrix

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "cx"

    def __str__(self):
        return "controlled-X gate"

    def inverse(self):
        _CX = CXGate(alias=None)
        _CX.cargs = copy.deepcopy(self.cargs)
        _CX.targs = copy.deepcopy(self.targs)
        return _CX

    def exec(self, circuit):
        exec_controlSingle(self, circuit)


CX = CXGate(["CX", "Cx", "cx"])


class CYGate(BasicGate):
    """ controlled-Y gate

    """

    _matrix = np.array([
        [0, -1j],
        [1j, 0]
    ], dtype=np.complex128)

    _compute_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, -1j],
        [0, 0, 1j, 0]
    ], dtype=np.complex128)

    @property
    def compute_matrix(self) -> np.ndarray:
        return self._compute_matrix

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "cy"

    def __str__(self):
        return "controlled-Y gate"

    def inverse(self):
        _CY = CYGate(alias=None)
        _CY.cargs = copy.deepcopy(self.cargs)
        _CY.targs = copy.deepcopy(self.targs)
        return _CY

    def exec(self, circuit):
        exec_controlSingle(self, circuit)


CY = CYGate(["CY", "Cy"])

class CHGate(BasicGate):
    """ controlled-Hadamard gate


    """

    _matrix = np.array([
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [1 / np.sqrt(2), -1 / np.sqrt(2)]
    ], dtype=np.complex128)

    _compute_matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
        [0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)]
    ], dtype=np.complex128)

    @property
    def compute_matrix(self) -> np.ndarray:
        return self._compute_matrix

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "ch"

    def __str__(self):
        return "controlled-Hadamard gate"

    def inverse(self):
        _CH = CHGate(alias=None)
        _CH.cargs = copy.deepcopy(self.cargs)
        _CH.targs = copy.deepcopy(self.targs)
        return _CH

    def exec(self, circuit):
        exec_controlSingle(self, circuit)


CH = CHGate(["CH", "Ch"])


class CRzGate(BasicGate):
    """ controlled-Rz gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 1
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "crz"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [np.exp(-self.parg / 2 * 1j), 0],
            [0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    @property
    def compute_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-self.parg / 2 * 1j), 0],
            [0, 0, 0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    def __str__(self):
        return "controlled-Rz gate"

    # @staticmethod
    # def type():
    #     return GateType.CRz

    def inverse(self):
        _CRz = CRzGate(alias=None)
        _CRz.cargs = copy.deepcopy(self.cargs)
        _CRz.targs = copy.deepcopy(self.targs)
        _CRz.pargs = [-self.pargs[0]]
        return _CRz

    def exec(self, circuit):
        exec_controlSingle(self, circuit)


CRz = CRzGate(["CRZ", "CRz", "Crz"])


class CU1Gate(BasicGate):
    """ Controlled-U1 gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 1
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "cu1"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [1, 0],
            [0, np.exp(1j * self.pargs[0])]
        ], dtype=np.complex128)

    @property
    def compute_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(1j * self.pargs[0])]
        ], dtype=np.complex128)

    def __str__(self):
        return "controlled-U1 gate"

    def inverse(self):
        _CU1 = CU1Gate(alias=None)
        _CU1.cargs = copy.deepcopy(self.cargs)
        _CU1.targs = copy.deepcopy(self.targs)
        _CU1.pargs = [-self.pargs[0]]
        return _CU1

    def exec(self, circuit):
        exec_controlSingle(self, circuit)


CU1 = CU1Gate(["CU1", "cu1"])


class CU3Gate(BasicGate):
    """ Controlled-U3 gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 1
        self.targets = 1
        self.params = 3
        self.pargs = [np.pi / 2, 0, 0]
        self.qasm_name = "cu3"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [np.cos(self.pargs[0] / 2), -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2)],
            [np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
             np.exp(1j * (self.pargs[1] + self.pargs[2])) * np.cos(self.pargs[0] / 2)]
        ], dtype=np.complex128)

    @property
    def compute_matrix(self) -> np.ndarray:
        return np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.cos(self.pargs[0] / 2), -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2)],
            [0, 0, np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
             np.exp(1j * (self.pargs[1] + self.pargs[2])) * np.cos(self.pargs[0] / 2)]
        ], dtype=np.complex128)

    def __str__(self):
        return "controlled-U3 gate"

    def inverse(self):
        _CU3 = CU3Gate(alias=None)
        _CU3.cargs = copy.deepcopy(self.cargs)
        _CU3.targs = copy.deepcopy(self.targs)
        _CU3.pargs = [self.pargs[0], np.pi - self.pargs[2], np.pi - self.pargs[1]]
        return _CU3

    def exec(self, circuit):
        exec_controlSingle(self, circuit)


CU3 = CU3Gate(["CU3", "cu3"])


class FSimGate(BasicGate):
    """ fSim gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 2
        self.params = 2
        self.pargs = [np.pi / 2, 0]
        self.qasm_name = "fsim"

    @property
    def matrix(self) -> np.ndarray:
        costh = np.cos(self.pargs[0])
        sinth = np.sin(self.pargs[0])
        phi = self.pargs[1]
        return np.array([
            [1, 0, 0, 0],
            [0, costh, -1j * sinth, 0],
            [0, -1j * sinth, costh, 0],
            [0, 0, 0, np.exp(-1j * phi)]
        ], dtype=np.complex128)

    def __str__(self):
        return "fSim gate"

    def inverse(self):
        _Fsim = FSimGate(alias=None)
        _Fsim.targs = copy.deepcopy(self.targs)
        _Fsim.pargs = [-self.pargs[0], -self.pargs[1]]
        return _Fsim

    def exec(self, circuit):
        exec_two(self, circuit)


FSim = FSimGate(["FSim", "fSim", "fsim"])


class RxxGate(BasicGate):
    """ Rxx gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 2
        self.params = 1
        self.pargs = [0]
        self.qasm_name = "Rxx"

    @property
    def matrix(self) -> np.ndarray:
        costh = np.cos(self.parg / 2)
        sinth = np.sin(self.parg / 2)
        return np.array([
            [costh, 0, 0, -1j * sinth],
            [0, costh, -1j * sinth, 0],
            [0, -1j * sinth, costh, 0],
            [-1j * sinth, 0, 0, costh]
        ], dtype=np.complex128)

    def __str__(self):
        return "Rxx gate"

    def inverse(self):
        _Rxx = RxxGate(alias=None)
        _Rxx.targs = copy.deepcopy(self.targs)
        _Rxx.pargs = [-self.parg]
        return _Rxx

    def exec(self, circuit):
        exec_two(self, circuit)


Rxx = RxxGate(["Rxx", "rxx", "RXX"])


class RyyGate(BasicGate):
    """ Ryy gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 2
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "Ryy"

    @property
    def matrix(self) -> np.ndarray:
        costh = np.cos(self.parg / 2)
        sinth = np.sin(self.parg / 2)
        return np.array([
            [costh, 0, 0, 1j * sinth],
            [0, costh, -1j * sinth, 0],
            [0, -1j * sinth, costh, 0],
            [1j * sinth, 0, 0, costh]
        ], dtype=np.complex128)

    def __str__(self):
        return "Ryy gate"

    def inverse(self):
        _Ryy = RyyGate(alias=None)
        _Ryy.targs = copy.deepcopy(self.targs)
        _Ryy.pargs = [-self.parg]
        return _Ryy

    def exec(self, circuit):
        exec_two(self, circuit)


Ryy = RyyGate(["Ryy", "ryy", "RYY"])


class RzzGate(BasicGate):
    """ Rzz gate

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 2
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "Rzz"

    @property
    def matrix(self) -> np.ndarray:
        expth = np.exp(0.5j * self.parg)
        sexpth = np.exp(-0.5j * self.parg)
        return np.array([
            [sexpth, 0, 0, 0],
            [0, expth, 0, 0],
            [0, 0, expth, 0],
            [0, 0, 0, sexpth]
        ], dtype=np.complex128)

    def __str__(self):
        return "Rzz gate"

    def inverse(self):
        _Rzz = RzzGate(alias=None)
        _Rzz.targs = copy.deepcopy(self.targs)
        _Rzz.pargs = [-self.parg]
        return _Rzz

    def exec(self, circuit):
        exec_two(self, circuit)


Rzz = RzzGate(["Rzz", "rzz", "RZZ"])

class MeasureGate(BasicGate):
    """ z-axis Measure gate

    Measure one qubit along z-axis.
    After acting on the qubit(circuit flush), the qubit get the value 0 or 1
    and the amplitude changed by the result.

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "measure"

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("try to get the matrix of measure gate")

    def __str__(self):
        return "Measure gate"

    def exec(self, circuit):
        exec_measure(self, circuit)


Measure = MeasureGate(["Measure"])


class ResetGate(BasicGate):
    """ Reset gate

    Reset the qubit into 0 state,
    which change the amplitude

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "reset"

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("try to get the matrix of reset gate")

    def __str__(self):
        return "Reset gate"

    def exec(self, circuit):
        exec_reset(self, circuit)


Reset = ResetGate(["Reset"])


class BarrierGate(BasicGate):
    """ Barrier gate

    In IBMQ, barrier gate forbid the optimization cross the gate,
    It is invalid in out circuit now.

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "barrier"

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("try to get the matrix of barrier gate")

    def __str__(self):
        return "Barrier gate"

    def exec(self, circuit):
        exec_barrier(self, circuit)


Barrier = BarrierGate(["Barrier"])


class SwapGate(BasicGate):
    """ Swap gate

    In the computation, it will not change the amplitude.
    Instead, it change the index of a Tangle.

    """

    _matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 2
        self.params = 0
        self.qasm_name = "swap"

    def __str__(self):
        return "Swap gate"

    def inverse(self):
        _swap = SwapGate(alias=None)
        _swap.targs = copy.deepcopy(self.targs)
        return _swap

    def exec(self, circuit):
        exec_swap(self, circuit)


Swap = SwapGate(["Swap"])


class PermGate(BasicGate):
    """ Permutation gate

    A special gate defined in our circuit,
    It can change an n-qubit qureg's amplitude by permutaion,
    the parameter is a 2^n list describes the permutation.

    """

    # life cycle
    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, params = None, name = None):
        """ pass permutation to the gate

        the length of permutaion must be 2^n,
        by which we can calculate the number of targets

        Args:
            params(list/tuple): the permutation parameters

        Returns:
            PermGate: the gate after filled by parameters
        """
        self.__temp_name = name
        self.pargs = []
        if not isinstance(params, list) or not isinstance(params, tuple):
            TypeException("list or tuple", params)
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

    @property
    def compute_matrix(self):
        return self.matrix

    def __str__(self):
        return "Permutation gate"

    # @staticmethod
    # def type():
    #     return GateType.Perm

    def inverse(self):
        _perm = PermGate(alias=None)
        _perm.targs = copy.deepcopy(self.targs)
        matrix = [0] * self.params
        i = 0
        for parg in self.pargs:
            matrix[parg] = i
            i += 1
        _perm.pargs = matrix
        _perm.params = self.params
        _perm.targets = self.targets
        return _perm

    def exec(self, circuit):
        exec_perm(self, circuit)


Perm = PermGate(["Perm"])


class ControlPermMulDetailGate(BasicGate):
    """ controlled-Permutation gate

    This gate is used to implement oracle in the order-finding algorithm

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 1
        self.targets = 0
        self.params = 0

    def __call__(self, params = None, name = None):
        """ pass parameters to the gate

        give parameters (a, N) to the gate

        Args:
            params(list/tuple): the oracle's parameters a and N

        Returns:
            ControlPermMulDetailGate: the gate after filled by parameters
        """
        self.__temp_name = name
        self.pargs = []
        if not isinstance(params, list) or not isinstance(params, tuple):
            TypeException("list or tuple", params)
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

    def __str__(self):
        return "controlled-Permutation gate"

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

        _ControlP = ControlPermMulDetailGate()
        _ControlP.targs = copy.deepcopy(self.targs)
        _ControlP.pargs = [n_inverse, self.pargs[1]]
        _ControlP.targets = self.targets
        _ControlP.params = self.params
        _ControlP.controls = self.controls
        return _ControlP

    def exec(self, circuit):
        exec_controlMulPerm(self, circuit)


ControlPermMulDetail = ControlPermMulDetailGate(["ControlPermMulDetail"])


class PermShiftGate(PermGate):
    """ act an increase or subtract operate with modulus.

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, params = None, N=None, name=None):
        """ pass parameters to the gate

        give parameters (params, N) to the gate

        Args:
            params(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermShiftGate: the gate after filled by parameters
        """
        self.__temp_name = name
        if not isinstance(params, int):
            raise TypeException("int", params)
        if N is None:
            raise Exception("PermShift need two parameters")
        if not isinstance(N, int):
            raise TypeException("int", N)

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


PermShift = PermShiftGate(["PermShift"])


class ControlPermShiftGate(PermGate):
    """ Controlled-PermShiftGate

    PermShiftGate with a control bit

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 1
        self.targets = 0
        self.params = 0

    def __call__(self, params = None, N=None, name=None):
        """ pass parameters to the gate

        give parameters (params, N) to the gate

        Args:
            params(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermShiftGate: the gate after filled by parameters
        """
        self.__temp_name = name
        if not isinstance(params, int):
            raise TypeException("int", params)
        if N is None:
            raise Exception("ControlPermShift need two parameters")
        if not isinstance(N, int):
            raise TypeException("int", N)

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


ControlPermShift = ControlPermShiftGate(["ControlPermShift"])


class PermMulGate(PermGate):
    """ act an multiply operate with modulus.

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, params = None, N=None, name=None):
        """ pass parameters to the gate

        give parameters (params, N) to the gate

        Args:
            params(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermMulGate: the gate after filled by parameters
        """
        self.__temp_name = name
        if not isinstance(params, int):
            raise TypeException("int", params)
        if N is None:
            raise Exception("PermMul need two parameters")
        if not isinstance(N, int):
            raise TypeException("int", N)
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


PermMul = PermMulGate(["PermMul"])


class ControlPermMulGate(PermGate):
    """ a controlled-PermMulGate


    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, params = None, N=None, name=None):
        """ pass parameters to the gate

        give parameters (params, N) to the gate

        Args:
            params(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            ControlPermMulGate: the gate after filled by parameters
        """
        self.__temp_name = name
        if not isinstance(params, int):
            raise TypeException("int", params)
        if N is None:
            raise Exception("PermMul need two parameters")
        if not isinstance(N, int):
            raise TypeException("int", N)
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


ControlPermMul = ControlPermMulGate(["ControlPermMul"])


class PermFxGate(PermGate):
    """ act an Fx oracle on a qureg

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, params = None, name = None):
        """ pass Fx to the gate

        Fx should be a 2^n list that represent a boolean function
        {0, 1}^n -> {0, 1}

        Args:
            params(list):contain 2^n values which are 0 or 1

        Returns:
            PermFxGate: the gate after filled by parameters
        """
        self.__temp_name = name
        if not isinstance(params, list):
            raise TypeException("list", params)
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

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.array([], dtype=np.complex128)
        for i in range(self.params):
            for j in range(self.params):
                if self.pargs[i] == j:
                    np.append(matrix, 1)
                else:
                    np.append(matrix, 0)
        return matrix


PermFx = PermFxGate(["PermFx"])


class UnitaryGate(BasicGate):
    """ Custom gate

    act an unitary matrix on the qureg,
    the parameters is the matrix

    """

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        self._matrix = matrix

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, params = None, name = None):
        """ pass the unitary matrix

        Args:
            params(np.array/list/tuple): contain 2^n * 2^n elements, which
            form an unitary matrix.


        Returns:
            UnitaryGateGate: the gate after filled by parameters
        """
        self.__temp_name = name
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
            raise TypeException("list or tuple", params)
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

    def __str__(self):
        return "Unitary gate"

    def inverse(self):
        _unitary = UnitaryGate()
        _unitary.targs = copy.deepcopy(self.targs)
        _unitary.matrix = np.array(np.mat(self._matrix).reshape(1 << self.targets, 1 << self.targets).
                                   H.reshape(1, -1), dtype=np.complex128)
        _unitary.targets = self.targets
        _unitary.params = self.params
        _unitary.controls = self.controls
        return _unitary

    def exec(self, circuit):
        exec_unitary(self, circuit)


Unitary = UnitaryGate(["Unitary"])


class ShorInitialGate(BasicGate):
    """ a oracle gate to preparation the initial state before IQFT in Shor algorithm

    backends will preparation the initial state by classical operator
    with a fixed measure result of second register.

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, params = None, name = None):
        """ pass the parameters

        Args:
            params(list/tuple): contain the parameters x, N and u which indicate
            the base number, exponential and the measure result of the second register.

        Returns:
            ShorInitialGate: the gate after filled by parameters

        """
        self.__temp_name = name
        if not isinstance(params, list) and not isinstance(params, tuple):
            raise TypeException("list or tuple", params)
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

    def exec(self, circuit):
        exec_shorInit(self, circuit)


ShorInitial = ShorInitialGate(["ShorInitial"])


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

    Note that all subClass must overloaded the function "build_gate"
    """

    def build_gate(self):
        """ generate BasicGate, affectArgs

        Returns:
            CompositeGate: synthetize result
        """
        affectArgs = self.affectArgs
        GateBuilder.setGateType(GATE_ID["X"])
        GateBuilder.setTargs(len(affectArgs) - 1)
        return CompositeGate(GateBuilder.getGate())

    def exec(self, circuit):
        gateSet = self.build_gate()
        for gate in gateSet:
            gate.exec(circuit)

class CCXGate(ComplexGate):
    """ Toffoli gate

    When using this gate, it will be showed as a whole gate
    instend of being split into smaller gate

    """

    _matrix = np.array([
        [0, 1],
        [1, 0]
    ], dtype=np.complex128)

    _compute_matrix = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0]
    ], dtype=np.complex128)

    @property
    def compute_matrix(self) -> np.ndarray:
        return self._compute_matrix

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 2
        self.targets = 1
        self.params = 0
        self.qasm_name = "ccx"

    def __str__(self):
        return "Toffoli gate"

    def inverse(self):
        _CCX = CCXGate(alias=None)
        _CCX.cargs = copy.deepcopy(self.cargs)
        _CCX.targs = copy.deepcopy(self.targs)
        return _CCX

    def build_gate(self):
        from .composite_gate import CompositeGate
        qureg = self.affectArgs
        gates = CompositeGate()

        with gates:
            H        & qureg[2]
            CX       & (qureg[2], qureg[1])
            T_dagger & qureg[1]
            CX       & (qureg[0], qureg[1])
            T        & qureg[1]
            CX       & (qureg[2], qureg[1])
            T_dagger & qureg[1]
            CX       & (qureg[0], qureg[1])
            T        & qureg[1]
            CX       & (qureg[0], qureg[2])
            T_dagger & qureg[2]
            CX       & (qureg[0], qureg[2])
            T        & qureg[0]
            T        & qureg[2]
            H        & qureg[2]
        return gates

    def exec(self, circuit):
        exec_toffoli(self, circuit)

CCX = CCXGate(["CCX", "CCx", "Ccx"])

class CCRzGate(ComplexGate):
    """ controlled-Rz gate with two control bits

    """

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 2
        self.targets = 1
        self.params = 1
        self.pargs = [0]
        self.qasm_name = "CCRz"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            [np.exp(-self.parg / 2 * 1j), 0],
            [0, np.exp(self.parg / 2 * 1j)]
        ], dtype=np.complex128)

    @property
    def compute_matrix(self) -> np.ndarray:
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

    def __str__(self):
        return "CCRz gate"

    def inverse(self):
        _CCRz = CCRzGate(alias=None)
        _CCRz.cargs = copy.deepcopy(self.cargs)
        _CCRz.targs = copy.deepcopy(self.targs)
        _CCRz.pargs = -self.parg
        return _CCRz

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

CCRz = CCRzGate(["CCRz"])

class QFTGate(ComplexGate):
    """ QFT gate

    act QFT on the qureg

    """

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            gateSet = self.build_gate()
            self._matrix = gateSet.matrix()
        return self._matrix

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self._matrix = None
        self.controls = 0
        self.targets = 3
        self.params = 0

    def __call__(self, params = None, name = None):
        """ pass the unitary matrix

        Args:
            params(int): point out the number of bits of the gate


        Returns:
            QFTGate: the QFTGate after filled by target number
        """
        self.params = params
        self.__temp_name = name
        return self

    def __str__(self):
        return "QFT gate"

    def inverse(self):
        _IQFT = IQFTGate()
        _IQFT.targs = copy.deepcopy(self.targs)
        _IQFT.targets = self.targets
        return _IQFT

    def build_gate(self):
        from .composite_gate import CompositeGate
        qureg = self.affectArgs
        gates = CompositeGate()

        with gates:
            for i in range(self.targets):
                H & qureg[i]
                for j in range(i + 1, self.targets):
                    CRz(2 * np.pi / (1 << j - i + 1)) & (qureg[j], qureg[i])
        return gates


QFT = QFTGate(["QFT", "qft"])


class IQFTGate(ComplexGate):
    """ IQFT gate

    act IQFT on the qureg

    """

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            gateSet = self.build_gate()
            self._matrix = gateSet.matrix()
        return self._matrix

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self._matrix = None
        self.controls = 0
        self.targets = 3
        self.params = 0

    def __call__(self, params = None, name = None):
        """ pass the unitary matrix

        Args:
            params(int): point out the number of bits of the gate


        Returns:
            IQFTGate: the IQFTGate after filled by target number
        """
        self.params = params
        self.__temp_name = name
        return self

    def __str__(self):
        return "QFT gate"

    def inverse(self):
        _QFT = QFTGate()
        _QFT.targs = copy.deepcopy(self.targs)
        _QFT.targets = self.targets
        return _QFT

    def build_gate(self):
        from .composite_gate import CompositeGate
        qureg = self.affectArgs
        gates = CompositeGate()

        with gates:
            for i in range(self.targets - 1, -1, -1):
                for j in range(self.targets - 1, i, -1):
                    CRz(-2 * np.pi / (1 << j - i + 1)) & (qureg[j], qureg[i])
                H & qureg[i]
        return gates


IQFT = IQFTGate(["IQFT", "iqft"])

class CSwapGate(ComplexGate):
    """ Fredkin gate

    When using this gate, it will be showed as a whole gate
    instend of being split into smaller gate

    """

    _matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128)

    _compute_matrix = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ], dtype=np.complex128)

    @property
    def compute_matrix(self) -> np.ndarray:
        return self._compute_matrix

    def __init__(self, alias=None):
        _add_alias(alias=alias, standard_name=self.__class__.__name__)
        super().__init__(alias=None)
        self.controls = 1
        self.targets = 2
        self.params = 0
        self.qasm_name = "cswap"

    def __str__(self):
        return "cswap gate"

    def inverse(self):
        _CSwap = CSwapGate(alias=None)
        _CSwap.cargs = copy.deepcopy(self.cargs)
        _CSwap.targs = copy.deepcopy(self.targs)
        return _CSwap

    def build_gate(self):
        from .composite_gate import CompositeGate
        qureg = self.affectArgs
        gates = CompositeGate()

        with gates:
            CX       & (qureg[2], qureg[1])
            H        & qureg[2]
            CX       & (qureg[2], qureg[1])
            T_dagger & qureg[1]
            CX       & (qureg[0], qureg[1])
            T        & qureg[1]
            CX       & (qureg[2], qureg[1])
            T_dagger & qureg[1]
            CX       & (qureg[0], qureg[1])
            T        & qureg[1]
            CX       & (qureg[0], qureg[2])
            T_dagger & qureg[2]
            CX       & (qureg[0], qureg[2])
            T        & qureg[0]
            T        & qureg[2]
            H        & qureg[2]
            CX       & (qureg[2], qureg[1])
        return gates

CSwap = CSwapGate(["Fredkin", "CSwap", "cswap"])
