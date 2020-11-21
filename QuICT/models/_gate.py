#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/10 9:04 下午
# @Author  : Han Yu
# @File    : _gate.py

from enum import Enum
import copy

import numpy as np

from QuICT.exception import TypeException
from ._qubit import Qubit, Qureg
from ._circuit import Circuit

class GateType(Enum):
    """ indicate the type of a basic gate

    Every Gate have a attribute named type, which indicate its type
    """
    Error = -1

    H = 0
    S = 1
    S_dagger = 2
    X = 3
    Y = 4
    Z = 5
    ID = 6
    U0 = 7
    U1 = 8
    U2 = 9
    U3 = 10
    Rx = 11
    Ry = 12
    Rz = 13
    T = 14
    T_dagger = 15
    CZ = 16
    CX = 17
    CY = 18
    CH = 19
    CRz = 20
    CCX = 21

    Measure = 22
    Reset = 23
    Barrier = 24

    Swap = 25

    Perm = 26
    Custom = 27
    ControlPermMulDetail = 28
    ShorInital = 29

class BasicGate(object):
    """ the abstract SuperClass of all basic quantum gates

    All basic quantum gates described in the framework have
    some common attributes and some common functions
    which defined in this class

    Attributes:
        controls(list<int>): the number of the control bits of the gate
        cargs(list<int>): the list of the index of control bits in the circuit
        carg(int, read only): the first object of cargs

        targets(list<int>): the number of the target bits of the gate
        targs(list<int>): the list of the index of target bits in the circuit
        targ(int, read only): the first object of targs

        params(list): the number of the parameter of the gate
        pargs(list): the list of the parameter
        prag(read only): the first object of pargs

        qasm_name(str, read only): gate's name in the OpenQASM 2.0
        type(GateType, read only): gate's type described by GateType

        matrix(np.array): the unitary matrix of the quantum gates act on targets
        computer_matrix(np.array): the unitary matrix of the quantum gates act on controls and targets
    """

    # Attribute
    @property
    def matrix(self) -> list:
        return self.__matrix

    @matrix.setter
    def matrix(self, matrix: list):
        self.__matrix = matrix

    def compute_matrix(self):
        return self.matrix

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

    @staticmethod
    def type():
        return GateType.Error

    # life cycle
    def __init__(self):
        self.__matrix = []
        self.__cargs = []
        self.__targs = []
        self.__pargs = []
        self.__controls = 0
        self.__targets = 0
        self.__params = 0
        self.__qasm_name = 'error'

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
                2) qureg
                3) tuple<qubit, qureg>
                4) list<qubit, qureg>
        Raise:
            TypeException: the type of other is wrong
        """

        if self.is_single() or self.is_measure() or self.is_barrier() or self.is_reset():
            # the gate is one qubit gate
            if isinstance(targets, tuple):
                for qubit in targets:
                    if not isinstance(qubit, Qubit):
                        raise TypeException("qubit或tuple<qubit, qureg>或qureg或list<qubit, qureg>或circuit", targets)
                    self._deal_qubit(qubit)
            elif isinstance(targets, Qubit):
                self._deal_qubit(targets)
            elif isinstance(targets, Qureg):
                for qubit in targets:
                    self._deal_qubit(qubit)
            elif isinstance(targets, Circuit):
                for qubit in targets.qubits:
                    self._deal_qubit(qubit)
            else:
                raise TypeException("qubit或tuple<qubit>或qureg或circuit", targets)
        else:
            # the gate is not one qubit gate
            if isinstance(targets, tuple):
                targets = list(targets)
            if isinstance(targets, list):
                qureg = Qureg()
                for item in targets:
                    if isinstance(item, Qubit):
                        qureg.append(item)
                    elif isinstance(item, Qureg):
                        qureg.extend(item)
                    else:
                        raise TypeException("qubit或tuple<qubit, qureg>或qureg或list<qubit, qureg>或circuit", targets)
            elif isinstance(targets, Qureg):
                qureg = targets
            elif isinstance(targets, Circuit):
                qureg = Qureg(targets.qubits)
            else:
                raise TypeException("qubit或tuple<qubit>或qureg或circuit", targets)
            self._deal_qureg(qureg)

    def __call__(self, params):
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
        if self.permit_element(params):
            self.pargs = [params]
        elif isinstance(params, list):
            self.pargs = []
            for element in params:
                if not self.permit_element(element):
                    raise TypeException("int或float或complex", element)
                self.pargs.append(element)
        elif isinstance(params, tuple):
            self.pargs = []
            for element in params:
                if not self.permit_element(element):
                    raise TypeException("int或float或complex", element)
                self.pargs.append(element)
        else:
            raise TypeException("int/float/complex或list<int/float/complex>或tuple<int/float/complex>", params)
        return self

    # get information of gate
    def print_info(self):
        """ print the information of the gate

        print the gate's information, including controls, targets and parameters

        """
        infomation = self.__str__()
        if self.controls != 0:
            infomation = infomation + f" 控制位:{self.cargs} "
        if self.targets != 0:
            infomation = infomation + f" 作用位:{self.targs} "
        if self.params != 0:
            infomation = infomation + f" 参数:{self.pargs} "
        print(infomation)

    def qasm(self):
        """ generator OpenQASM string for the gate

        Return:
            string: the OpenQASM 2.0 describe of the gate
        """
        if self.qasm_name == 'error':
            return 'error'
        string = self.qasm_name
        if self.params > 0:
            string += '('
            for i in range(len(self.pargs)):
                if i != 0:
                    string += ', '
                string += str(self.pargs[i])
            string += ')'
        string += ' '
        first_in = True
        for p in self.cargs:
            if not first_in:
                string += ', '
            else:
                first_in = False
            string += f'q[{p}]'
        for p in self.targs:
            if not first_in:
                string += ', '
            else:
                first_in = False
            string += f'q[{p}]'
        string += ';\n'
        return string

    def inverse(self):
        """ the inverse of the gate

        Return:
            BasicGate: the inverse of the gate
        """
        raise Exception("未定义的逆")

    # gate information
    def is_single(self) -> bool:
        """ judge whether gate is a one qubit gate(excluding special gate like measure, reset, custom and so on)

        Returns:
            bool: True if it is a one qubit gate
        """
        return 0 <= self.type().value  <= 15

    def is_control_single(self) -> bool:
        """ judge whether gate has one control bit and one target bit

        Returns:
            bool: True if it is has one control bit and one target bit
        """
        return (self.type().value >= 16) and (self.type().value <= 20)

    def is_diagonal(self) -> bool:
        """ judge whether gate's matrix is diagonal

        Raise:
            gate must be basic one qubit gate or two qubit gate

        Returns:
            bool: True if gate's matrix is diagonal
        """
        if not self.is_single() and not self.is_control_single() and not self.is_swap():
            raise Exception("只考虑单比特门和基础双比特门")
        if self.is_single():
            matrix = self.matrix
            if abs(matrix[1]) < 1e-10 and abs(matrix[2]) < 1e-10:
                return True
            else:
                return False
        elif self.is_control_single():
            matrix = self.matrix
            if abs(matrix[1]) < 1e-10 and abs(matrix[2]) < 1e-10:
                return True
            else:
                return False
        else:
            return False

    def is_ccx(self) -> bool:
        """ judge whether gate is toffoli gate

        Returns:
            bool: True if gate is toffoli gate
        """
        return self.type() == GateType.CCX

    def is_measure(self) -> bool:
        """ judge whether gate is measure gate

        Returns:
            bool: True if gate is measure gate
        """
        return self.type() == GateType.Measure

    def is_reset(self) -> bool:
        """ judge whether gate is reset gate

        Returns:
            bool: True if gate is reset gate
        """
        return self.type() == GateType.Reset

    def is_swap(self) -> bool:
        """ judge whether gate is swap gate

        Returns:
            bool: True if gate is swap gate
        """
        return self.type() == GateType.Swap

    def is_perm(self) -> bool:
        """ judge whether gate is permutation gate

        Returns:
            bool: True if gate is permutation gate
        """
        return self.type() == GateType.Perm

    def is_custom(self) -> bool:
        """ judge whether gate is custom gate

            Returns:
                bool: True if gate is custom gate
        """
        return self.type() == GateType.Custom

    def is_shorInit(self) -> bool:
        """ judge whether gate is ShorInit gate

        Returns:
            bool: True if gate is ShorInit gate
        """
        return self.type() == GateType.ShorInital

    def is_controlMulPer(self) -> bool:
        """ judge whether gate is ControlPermMulDetail gate

        Returns:
            bool: True if gate is ControlPermMulDetail gate
        """
        return self.type() == GateType.ControlPermMulDetail

    def is_barrier(self) -> bool:
        """ judge whether gate is Barrier gate

        Returns:
            bool: True if gate is Barrier gate
        """
        return self.type() == GateType.Barrier

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
            return False

    # private tool function
    def _deal_qubit(self, qubit):
        """ add gate to one qubit

        Args:
            qubit: qubit the gate act on

        """
        name = str(self.__class__.__name__)
        gate = globals()[name]()
        gate.pargs = copy.deepcopy(self.pargs)
        gate.targets = self.targets
        gate.controls = self.controls
        gate.params = self.params
        qubit.circuit.__add_qubit_gate__(gate, qubit)

    def _deal_qureg(self, qureg):
        """ add gate to one qureg

        Args:
            qureg: qureg the gate act on

        """
        name = str(self.__class__.__name__)
        gate = globals()[name]()
        gate.pargs = copy.deepcopy(self.pargs)
        gate.targets = self.targets
        gate.controls = self.controls
        gate.params = self.params
        if isinstance(gate, CustomGate):
            gate.matrix = self.matrix
        qureg.circuit.__add_qureg_gate__(gate, qureg)


class HGate(BasicGate):
    """ Hadamard gate


    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "h"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1 / np.sqrt(2), 1 / np.sqrt(2),
            1 / np.sqrt(2), -1 / np.sqrt(2)
        ], dtype = np.complex)

    def __str__(self):
        return "H门"

    @staticmethod
    def type():
        """
        :return: 返回H
        """
        return GateType.H

    def inverse(self):
        _H = HGate()
        _H.targs = copy.deepcopy(self.targs)
        return _H

H = HGate()

class SGate(BasicGate):
    """ Phase gate


    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "s"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1, 0,
            0, 1j
        ], dtype = np.complex)

    def __str__(self):
        return "Phase gate"

    @staticmethod
    def type():
        return GateType.S

    def inverse(self):
        _S_dagger = SDaggerGate()
        _S_dagger.targs = copy.deepcopy(self.targs)
        return _S_dagger

S = SGate()


class SDaggerGate(BasicGate):
    """ The conjugate transpose of Phase gate


    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "sdg"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1, 0,
            0, -1j
        ], dtype = np.complex)

    def __str__(self):
        return "The conjugate transpose of Phase gate"

    @staticmethod
    def type():
        return GateType.S_dagger

    def inverse(self):
        _SBACK = SGate()
        _SBACK.targs = copy.deepcopy(self.targs)
        return _SBACK

S_dagger = SDaggerGate()


class XGate(BasicGate):
    """ Pauli-X gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "x"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            0, 1,
            1, 0
        ], dtype = np.complex)

    def __str__(self):
        return "Pauli-X gate"

    @staticmethod
    def type():
        return GateType.X

    def inverse(self):
        _X = XGate()
        _X.targs = copy.deepcopy(self.targs)
        return _X

X = XGate()


class YGate(BasicGate):
    """ Pauli-Y gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "y"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            0, -1j,
            1j, 0
        ], dtype = np.complex)

    def __str__(self):
        return "Pauli-Y gate"

    @staticmethod
    def type():
        return GateType.Y

    def inverse(self):
        _Y = YGate()
        _Y.targs = copy.deepcopy(self.targs)
        return _Y

Y = YGate()


class ZGate(BasicGate):
    """ Pauli-Z gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "z"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1, 0,
            0, -1
        ], dtype = np.complex)

    def __str__(self):
        return "Pauli-Z gate"

    @staticmethod
    def type():
        return GateType.Z

    def inverse(self):
        _Z = ZGate()
        _Z.targs = copy.deepcopy(self.targs)
        return _Z

Z = ZGate()


class IDGate(BasicGate):
    """ Identity gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "id"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1, 0,
            0, 1
        ], dtype = np.complex)

    def __str__(self):
        return "Identity gate"

    @staticmethod
    def type():
        return GateType.ID

    def inverse(self):
        _ID = IDGate()
        _ID.targs = copy.deepcopy(self.targs)
        return _ID


ID = IDGate()

class U1Gate(BasicGate):
    """ Diagonal single-qubit gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "u1"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1, 0,
            0, np.exp(1j * self.pargs[0])
        ], dtype = np.complex)

    def __str__(self):
        return "U1 gate"

    @staticmethod
    def type():
        return GateType.U1

    def inverse(self):
        _U1 = U1Gate()
        _U1.targs = copy.deepcopy(self.targs)
        _U1.pargs = [-self.pargs[0]]
        return _U1

U1 = U1Gate()

class U2Gate(BasicGate):
    """ One-pulse single-qubit gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 2
        self.pargs = [np.pi / 2, np.pi / 2]
        self.qasm_name = "u2"

    @property
    def matrix(self) -> np.ndarray:
        sqrt2 = 1 / np.sqrt(2)
        return np.array([
            1 * sqrt2,
            -np.exp(1j * self.pargs[1]) * sqrt2,
            np.exp(1j * self.pargs[0]) * sqrt2,
            np.exp(1j * (self.pargs[0] + self.pargs[1])) * sqrt2
        ], dtype = np.complex)

    def __str__(self):
        return "U2 gate"

    @staticmethod
    def type():
        return GateType.U2

    def inverse(self):
        _U2 = U2Gate()
        _U2.targs = copy.deepcopy(self.targs)
        _U2.pargs = [np.pi - self.pargs[1], np.pi - self.pargs[0]]
        return _U2

U2 = U2Gate()

class U3Gate(BasicGate):
    """ Two-pulse single-qubit gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 3
        self.pargs = [0, 0, np.pi / 2]
        self.qasm_name = "u3"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            np.cos(self.pargs[0] / 2),
            -np.exp(1j * self.pargs[2]) * np.sin(self.pargs[0] / 2),
            np.exp(1j * self.pargs[1]) * np.sin(self.pargs[0] / 2),
            np.exp(1j * (self.pargs[1] + self.pargs[2])) * np.cos(self.pargs[0] / 2)
        ], dtype = np.complex)

    def __str__(self):
        return "U3 gate"

    @staticmethod
    def type():
        return GateType.U3

    def inverse(self):
        _U3 = U3Gate()
        _U3.targs = copy.deepcopy(self.targs)
        _U3.pargs = [self.pargs[0], np.pi - self.pargs[2], np.pi - self.pargs[1]]
        return _U3

U3 = U3Gate()

class RxGate(BasicGate):
    """ Rotation around the x-axis gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "rx"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            np.cos(self.parg / 2),
            1j * -np.sin(self.parg / 2),
            1j * -np.sin(self.parg / 2),
            np.cos(self.parg / 2),
        ], dtype = np.complex)

    def __str__(self):
        return "Rx gate"

    @staticmethod
    def type():
        return GateType.Rx

    def inverse(self):
        _Rx = RxGate()
        _Rx.targs = copy.deepcopy(self.targs)
        _Rx.pargs = [-self.pargs[0]]
        return _Rx

Rx = RxGate()

class RyGate(BasicGate):
    """ Rotation around the y-axis gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "ry"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            np.cos(self.pargs[0] / 2),
            -np.sin(self.pargs[0] / 2),
            np.sin(self.pargs[0] / 2),
            np.cos(self.pargs[0] / 2),
        ], dtype = np.complex)

    def __str__(self):
        return "Ry gate"

    @staticmethod
    def type():
        return GateType.Ry

    def inverse(self):
        _Ry = RyGate()
        _Ry.targs = copy.deepcopy(self.targs)
        _Ry.pargs = [-self.pargs[0]]
        return _Ry

Ry = RyGate()

class RzGate(BasicGate):
    """ Rotation around the z-axis gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 1
        self.pargs = [np.pi / 2]
        self.qasm_name = "rz"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,
            0,
            0,
            np.cos(self.pargs) + 1j * np.sin(self.pargs)
        ], dtype = np.complex)

    def __str__(self):
        return "Rz gate"

    @staticmethod
    def type():
        return GateType.Rz

    def inverse(self):
        _Rz = RzGate()
        _Rz.targs = copy.deepcopy(self.targs)
        _Rz.pargs = [-self.pargs[0]]
        return _Rz

Rz = RzGate()

class TGate(BasicGate):
    """ T gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "t"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,  0,
            0,  1 / np.sqrt(2) + 1j * 1 / np.sqrt(2)
        ], dtype = np.complex)

    def __str__(self):
        return "T gate"

    @staticmethod
    def type():
        return GateType.T

    def inverse(self):
        _Tdagger = TDaggerGate()
        _Tdagger.targs = copy.deepcopy(self.targs)
        return _Tdagger

T = TGate()

class TDaggerGate(BasicGate):
    """ The conjugate transpose of T gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "tdg"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,  0,
            0,  1 / np.sqrt(2) + 1j * -1 / np.sqrt(2)
        ], dtype = np.complex)

    def __str__(self):
        return "The conjugate transpose of T gate"

    @staticmethod
    def type():
        return GateType.T_dagger

    def inverse(self):
        _Tgate = TGate()
        _Tgate.targs = copy.deepcopy(self.targs)
        return _Tgate

T_dagger = TDaggerGate()

class CZGate(BasicGate):
    """ controlled-Z gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "cz"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,  0,
            0,  -1
        ], dtype = np.complex)

    def compute_matrix(self):
        return np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, -1
        ], dtype = np.complex)

    def __str__(self):
        return "controlled-Z gate"

    @staticmethod
    def type():
        return GateType.CZ

    def inverse(self):
        _CZ = CZGate()
        _CZ.cargs = copy.deepcopy(self.cargs)
        _CZ.targs = copy.deepcopy(self.targs)
        return _CZ

CZ = CZGate()

class CXGate(BasicGate):
    """ "controlled-X gate"


    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "cx"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            0,  1,
            1,  0
        ], dtype = np.complex)

    def compute_matrix(self):
        return np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 1,
            0, 0, 1, 0
        ], dtype = np.complex)

    def __str__(self):
        return "controlled-X gate"

    @staticmethod
    def type():
        return GateType.CX

    def inverse(self):
        _CX = CXGate()
        _CX.cargs = copy.deepcopy(self.cargs)
        _CX.targs = copy.deepcopy(self.targs)
        return _CX

CX = CXGate()

class CYGate(BasicGate):
    """ controlled-Y gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "cy"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            0,  1,
            1,  0
        ], dtype = np.complex)

    def compute_matrix(self):
        return np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 0, 1,
            0, 0, 1, 0
        ], dtype = np.complex)

    def __str__(self):
        return "controlled-Y gate"

    @staticmethod
    def type():
        return GateType.CY

    def inverse(self):
        _CY = CYGate()
        _CY.cargs = copy.deepcopy(self.cargs)
        _CY.targs = copy.deepcopy(self.targs)
        return _CY

CY = CYGate()

class CHGate(BasicGate):
    """ controlled-Hadamard gate


    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 1
        self.params = 0
        self.qasm_name = "ch"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1 / np.sqrt(2), 1 / np.sqrt(2),
            1 / np.sqrt(2), -1 / np.sqrt(2)
        ], dtype = np.complex)

    def compute_matrix(self):
        return np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1 / np.sqrt(2), 1 / np.sqrt(2),
            0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2)
        ], dtype = np.complex)

    def __str__(self):
        return "controlled-Hadamard gate"

    @staticmethod
    def type():
        return GateType.CH

    def inverse(self):
        _CH = CHGate()
        _CH.cargs = copy.deepcopy(self.cargs)
        _CH.targs = copy.deepcopy(self.targs)
        return _CH

CH = CHGate()

class CRzGate(BasicGate):
    """ controlled-Rz gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 1
        self.params = 1
        self.qasm_name = "crz"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,
            0,
            0,
            np.cos(self.parg) + 1j * np.sin(self.parg)
        ], dtype = np.complex)

    def compute_matrix(self):
        return np.array([
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, np.cos(self.pargs[0]) + 1j * np.sin(self.pargs[0])
        ], dtype = np.complex)

    def __str__(self):
        return "controlled-Rz gate"

    @staticmethod
    def type():
        return GateType.CRz

    def inverse(self):
        _CRz = CRzGate()
        _CRz.cargs = copy.deepcopy(self.cargs)
        _CRz.targs = copy.deepcopy(self.targs)
        _CRz.pargs = [-self.pargs[0]]
        return _CRz

CRz = CRzGate()

class CCXGate(BasicGate):
    """ Toffoli gate

    When using this gate, it will be showed as a whole gate
    instend of being split into smaller gate

    """

    def __init__(self):
        super().__init__()
        self.controls = 2
        self.targets = 1
        self.params = 0
        self.qasm_name = "ccx"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            0,  1,
            1,  0
        ], dtype = np.complex)

    def compute_matrix(self) -> np.ndarray:
        return np.array([
            1,  0,  0,  0,  0,  0,  0,  0,
            0,  1,  0,  0,  0,  0,  0,  0,
            0,  0,  1,  0,  0,  0,  0,  0,
            0,  0,  0,  1,  0,  0,  0,  0,
            0,  0,  0,  0,  1,  0,  0,  0,
            0,  0,  0,  0,  0,  1,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  1,
            0,  0,  0,  0,  0,  0,  1,  0
        ], dtype = np.complex)

    def __str__(self):
        return "Toffoli gate"

    @staticmethod
    def type():
        return GateType.CCX

    def inverse(self):
        _CCX = CCXGate()
        _CCX.cargs = copy.deepcopy(self.cargs)
        _CCX.targs = copy.deepcopy(self.targs)
        return _CCX

CCX = CCXGate()

class MeasureGate(BasicGate):
    """ z-axis Measure gate

    Measure one qubit along z-axis.
    After acting on the qubit(circuit flush), the qubit get the value 0 or 1
    and the amplitude changed by the result.

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "measure"

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("try to get the matrix of measure gate")

    def __str__(self):
        return "Measure gate"

    @staticmethod
    def type():
        return GateType.Measure
Measure = MeasureGate()

class ResetGate(BasicGate):
    """ Reset gate

    Reset the qubit into 0 state,
    which change the amplitude

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "reset"

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("try to get the matrix of reset gate")

    def __str__(self):
        return "Reset gate"

    @staticmethod
    def type():
        return GateType.Reset

Reset = ResetGate()

class BarrierGate(BasicGate):
    """ Barrier gate

    In IBMQ, barrier gate forbid the optimization cross the gate,
    It is invalid in out circuit now.

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 1
        self.params = 0
        self.qasm_name = "barrier"

    @property
    def matrix(self) -> np.ndarray:
        raise Exception("try to get the matrix of barrier gate")

    def __str__(self):
        return "Barrier gate"

    @staticmethod
    def type():
        return GateType.Barrier

Barrier = BarrierGate()

class SwapGate(BasicGate):
    """ Swap gate

    In the computation, it will not change the amplitude.
    Instead, it change the index of a Tangle.

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 2
        self.params = 0
        self.qasm_name = "swap"

    @property
    def matrix(self) -> np.ndarray:
        return np.array([
            1,  0,  0,  0,
            0,  0,  1,  0,
            0,  1,  0,  0,
            0,  0,  0,  1
        ], dtype = np.complex)

    def __str__(self):
        return "Swap gate"

    @staticmethod
    def type():
        return GateType.Swap

    def inverse(self):
        _swap = SwapGate()
        _swap.targs = copy.deepcopy(self.targs)
        return _swap

Swap = SwapGate()

class PermGate(BasicGate):
    """ Permutation gate

    A special gate defined in our circuit,
    It can change an n-qubit qureg's amplitude by permutaion,
    the parameter is a 2^n list describes the permutation.

    """

    # life cycle
    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, permutation):
        """ pass permutation to the gate

        the length of permutaion must be 2^n,
        by which we can calculate the number of targets

        Args:
            permutation(list/tuple): the permutation parameters

        Returns:
            PermGate: the gate after filled by parameters
        """
        self.pargs = []
        if not isinstance(permutation, list) or not isinstance(permutation, tuple):
            TypeException("list or tuple", permutation)
        if isinstance(permutation, tuple):
            permutation = list(permutation)
        length = len(permutation)
        if length == 0:
            raise Exception("list or tuple shouldn't be empty")
        n = int(round(np.log2(length)))
        if (1 << n) != length:
            raise Exception("the length of list or tuple should be the power of 2")
        self.params = length
        self.targets = n
        for idx in permutation:
            if not isinstance(idx, int) or idx < 0 or idx >= self.params:
                raise Exception("the element in the list/tuple should be integer")
            if idx in self.pargs:
                raise Exception("the list/tuple should be a permutation for [0, 2^n) without repeat")
            self.pargs.append(idx)
        return self

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.array([], dtype = np.complex)
        for i in range(self.params):
            for j in range(self.params):
                if self.pargs[i] == j:
                    matrix = np.append(matrix, 1)
                else:
                    matrix = np.append(matrix, 0)
        return matrix

    def __str__(self):
        return "Permutation gate"

    @staticmethod
    def type():
        return GateType.Perm

    def inverse(self):
        _perm = PermGate()
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

Perm = PermGate()

class ControlPermMulDetailGate(BasicGate):
    """ controlled-Permutation gate

    This gate is used to implement oracle in the order-finding algorithm

    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 0
        self.params = 0

    def __call__(self, params):
        """ pass parameters to the gate

        give parameters (a, N) to the gate

        Args:
            params(list/tuple): the oracle's parameters a and N

        Returns:
            ControlPermMulDetailGate: the gate after filled by parameters
        """

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
        matrix = np.array([], dtype = np.complex)
        a = self.pargs[0]
        N = self.pargs[1]
        for idx in range(1 << self.targets):
            idxx = idx // 2
            controlxx = idx % 2
            if controlxx == 0:
                self.pargs.append(idx)
            else:
                if idxx >= N:
                    self.pargs.append(idx)
                else:
                    self.pargs.append(((idxx * a % N) << 1) + controlxx)
        return matrix

    def __str__(self):
        return "controlled-Permutation gate"

    @staticmethod
    def type():
        return GateType.ControlPermMulDetail
ControlPermMulDetail = ControlPermMulDetailGate()

class PermShiftGate(PermGate):
    """ act an increase or subtract operate with modulus.

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, shift, N = None):
        """ pass parameters to the gate

        give parameters (shift, N) to the gate

        Args:
            shift(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermShiftGate: the gate after filled by parameters
        """
        if not isinstance(shift, int):
            raise TypeException("int", shift)
        if N is None:
            raise Exception("PermShift need two parameters")
        if not isinstance(N, int):
            raise TypeException("int", N)

        if N <= 0:
            raise Exception("the modulus should be integer")
        n = int(round(np.log2(N)))
        self.params = N
        self.targets = n
        for idx in range(1 << self.targets):
            idxx = idx // 2
            controlxx = idx % 2
            if controlxx == 0:
                self.pargs.append(idx)
            else:
                if idxx < N:
                    self.pargs.append(idx)
                else:
                    self.pargs.append(((((idxx + shift) % N + N) % N) << 1) + controlxx)
        return self

PermShift = PermShiftGate()

class ControlPermShiftGate(PermGate):
    """ Controlled-PermShiftGate

    PermShiftGate with a control bit

    """

    def __init__(self):
        super().__init__()
        self.controls = 1
        self.targets = 0
        self.params = 0

    def __call__(self, shift, N = None):
        """ pass parameters to the gate

        give parameters (shift, N) to the gate

        Args:
            shift(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermShiftGate: the gate after filled by parameters
        """
        if not isinstance(shift, int):
            raise TypeException("int", shift)
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
                    self.pargs.append(((((idxx + shift) % N + N) % N) << 1) + controlxx)
        return self

ControlPermShift = ControlPermShiftGate()

class PermMulGate(PermGate):
    """ act an multiply operate with modulus.

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, shift, N = None):
        """ pass parameters to the gate

        give parameters (shift, N) to the gate

        Args:
            shift(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            PermMulGate: the gate after filled by parameters
        """
        if not isinstance(shift, int):
            raise TypeException("int", shift)
        if N is None:
            raise Exception("PermMul need two parameters")
        if not isinstance(N, int):
            raise TypeException("int", N)
        if N <= 0:
            raise Exception("the modulus should be integer")
        if shift <= 0:
            raise Exception("the shift should be integer")

        if np.gcd(shift, N) != 1:
            raise Exception(f"shift and N should be co-prime, but {shift} and {N} are not.")

        shift = shift % N

        n = int(round(np.log2(N)))
        if (1 << n) < N:
            n = n + 1
        self.params = N
        self.targets = n
        for idx in range(N):
            self.pargs.append(idx * shift % N)
        for idx in range(N, 1 << n):
            self.pargs.append(idx)
        return self

PermMul = PermMulGate()

class ControlPermMulGate(PermGate):
    """ a controlled-PermMulGate


    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, shift, N = None):
        """ pass parameters to the gate

        give parameters (shift, N) to the gate

        Args:
            shift(int): the number (can be negative) the qureg increase
            N(int): the modulus

        Returns:
            ControlPermMulGate: the gate after filled by parameters
        """
        if not isinstance(shift, int):
            raise TypeException("int", shift)
        if N is None:
            raise Exception("PermMul need two parameters")
        if not isinstance(N, int):
            raise TypeException("int", N)
        if N <= 0:
            raise Exception("the modulus should be integer")
        if shift <= 0:
            raise Exception("the shift should be integer")

        if np.gcd(shift, N) != 1:
            raise Exception(f"shift and N should be co-prime, but {shift} and {N} are not.")

        shift = shift % N

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
                    self.pargs.append(((idxx * shift % N) << 1) + controlxx)
        return self

ControlPermMul = ControlPermMulGate()

class PermFxGate(PermGate):
    """ act an Fx oracle on a qureg

    This Class is the subClass of PermGate.
    In fact, we calculate the permutation by the parameters.

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, f):
        """ pass Fx to the gate

        Fx should be a 2^n list that represent a boolean function
        {0, 1}^n -> {0, 1}

        Args:
            f(list):contain 2^n values which are 0 or 1

        Returns:
            PermFxGate: the gate after filled by parameters
        """
        if not isinstance(f, list):
            raise TypeException("list", f)
        n = int(round(np.log2(len(f))))
        if len(f) != 1 << n:
            raise Exception("the length of f should be the power of 2")
        N = 1 << n
        for i in range(N):
            if f[i] != 0 and f[i] != 1:
                raise Exception("the range of f should be {0, 1}")

        self.params = 1 << (n + 1)
        self.targets = n + 1

        N_2 = N << 1
        for idx in range(N_2):
            if f[idx & (N - 1)] == 1:
                self.pargs.append(idx ^ N)
            else:
                self.pargs.append(idx)
        return self

    @property
    def matrix(self) -> np.ndarray:
        matrix = np.array([], dtype = np.complex)
        for i in range(self.params):
            for j in range(self.params):
                if self.pargs[i] == j:
                    np.append(matrix, 1)
                else:
                    np.append(matrix, 0)
        return matrix

    @staticmethod
    def type():
        return GateType.Perm
PermFx = PermFxGate()

class CustomGate(BasicGate):
    """ Custom gate

    act an unitary matrix on the qureg,
    the parameters is the matrix

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, matrix):
        """ pass the unitary matrix

        Args:
            matrix(list/tuple): contain 2^n * 2^n elements, which
            form an unitary matrix.


        Returns:
            CustomGate: the gate after filled by parameters
        """
        if not isinstance(matrix, list) and not isinstance(matrix, tuple):
            raise TypeException("list or tuple", matrix)
        if isinstance(matrix, tuple):
            matrix = list(matrix)
        length = len(matrix)
        if length == 0:
            raise Exception("the list or tuple passed in shouldn't be empty")
        n2 = int(round(np.sqrt(length)))
        if n2 * n2 != length:
            raise Exception("the length of list or tuple should be the square of power(2, n)")
        n = int(round(np.log2(n2)))
        if (1 << n) != n2:
            raise Exception("the length of list or tuple should be the square of power(2, n)")
        self.targets = n
        self.matrix = np.array([matrix], dtype = np.complex)
        return self

    def __str__(self):
        return "Custom gate"

    @staticmethod
    def type():
        return GateType.Custom

    def inverse(self):
        _custom = CustomGate()
        _custom.targs = copy.deepcopy(self.targs)
        _custom.matrix = np.array(np.mat(self.matrix).reshape(1 << self.targets, 1 << self.targets).H.reshape(1, -1),
                                  dtype = np.complex)
        _custom.targets = self.targets
        _custom.params = self.params
        _custom.controls = self.controls
        return _custom

Custom = CustomGate()

class ShorInitialGate(BasicGate):
    """ a oracle gate to preparation the initial state before IQFT in Shor algorithm

    backends will preparation the initial state by classical operator
    with a fixed measure result of second register.

    """

    def __init__(self):
        super().__init__()
        self.controls = 0
        self.targets = 0
        self.params = 0

    def __call__(self, other):
        """ pass the parameters

        Args:
            other(list/tuple): contain the parameters x, N and u which indicate
            the base number, exponential and the measure result of the second register.

        Returns:
            ShorInitialGate: the gate after filled by parameters

        """
        if not isinstance(other, list) and not isinstance(other, tuple):
            raise TypeException("list or tuple", other)
        if isinstance(other, tuple):
            other = list(other)
        length = len(other)
        if length != 3:
            raise Exception("list or tuple passed in should contain three values")
        x = other[0]
        N = other[1]
        u = other[2]
        n = 2 * int(np.ceil(np.log2(N)))
        self.targets = n
        self.pargs = [x, N, u]
        return self

    @staticmethod
    def type():
        return GateType.ShorInital

ShorInitial = ShorInitialGate()

class GateBuilderModel(object):
    """ A model that help users get gate without circuit

    The model is designed to help users get some gates independent of the circuit
    Because there is no clear API to setting a gate's control bit indexes and
    target bit indexes without circuit or qureg.

    Users should set the gateType of the GateBuilder, than set necessary parameters
    (Targs, Cargs, Pargs). After that, user can get a gate from GateBuilder.

    """

    def __init__(self):
        self.gateType = GateType.Error
        self.pargs = []
        self.cargs = []
        self.targs = []

    def setGateType(self, type):
        """ pass the gateType for the builder

        Args:
            type(GateType): the type passed in
        """

        self.gateType = type

    def setTargs(self, targs):
        """ pass the target bits' indexes of the gate

        The targets should be passed.

        Args:
            targs(list/int/float/complex): the target bits' indexes the gate act on.
        """

        if isinstance(targs, list):
            self.targs = targs
        else:
            self.targs = [targs]

    def setCargs(self, cargs):
        """ pass the control bits' indexes of the gate

        if the gate don't need the control bits, needn't to call this function.

        Args:
            cargs(list/int/float/complex): the control bits' indexes the gate act on.
        """
        if isinstance(cargs, list):
            self.cargs = cargs
        else:
            self.cargs = [cargs]

    def setPargs(self, pargs):
        """ pass the parameters of the gate

        if the gate don't need the parameters, needn't to call this function.

        Args:
            pargs(list/int/float/complex): the parameters filled in the gate
        """

        if isinstance(pargs, list):
            self.pargs = pargs
        else:
            self.pargs = [pargs]

    def setArgs(self, args):
        """ pass the bits' indexed of the gate by one time

        The control bits' indexed first, and followed the targets bits' indexed.

        Args:
            args(list/int/float/complex): the act bits' indexes of the gate
        """

        if isinstance(args, list):
            if self.getCargsNumber() > 0:
                self.setCargs(args[0:self.getCargsNumber()])
            if self.getTargsNumber() > 0:
                self.setTargs(args[self.getCargsNumber():])
        else:
            self.setTargs([args])

    def getCargsNumber(self):
        """ get the number of cargs of the gate

        once the gateType is set, the function is valid.

        Return:
            int: the number of cargs
        """
        gate = self._inner_generate_gate()
        return gate.controls

    def getTargsNumber(self):
        """ get the number of targs of the gate

        once the gateType is set, the function is valid.

        Return:
            int: the number of targs
        """

        gate = self._inner_generate_gate()
        return gate.targets

    def getParamsNumber(self):
        """ get the number of pargs of the gate

        once the gateType is set, the function is valid.

        Return:
            int: the number of pargs
        """

        gate = self._inner_generate_gate()
        return gate.params

    def getGate(self):
        """ get the gate

        once the parameters are set, the function is valid.

        Return:
            BasicGate: the gate with parameters set in the builder
        """
        gate = self._inner_generate_gate()
        return self._inner_complete_gate(gate)

    def _inner_generate_gate(self):
        """ private tool function

        get an initial gate by the gateType set for builder

        Return:
            BasicGate: the initial gate
        """
        if self.gateType == GateType.H:
            return HGate()
        elif self.gateType == GateType.S:
            return SGate()
        elif self.gateType == GateType.S_dagger:
            return SDaggerGate()
        elif self.gateType == GateType.X:
            return XGate()
        elif self.gateType == GateType.Y:
            return YGate()
        elif self.gateType == GateType.Z:
            return ZGate()
        elif self.gateType == GateType.ID:
            return IDGate()
        elif self.gateType == GateType.U1:
            return U1Gate()
        elif self.gateType == GateType.U2:
            return U2Gate()
        elif self.gateType == GateType.U3:
            return U3Gate()
        elif self.gateType == GateType.Rx:
            return RxGate()
        elif self.gateType == GateType.Ry:
            return RyGate()
        elif self.gateType == GateType.Rz:
            return RzGate()
        elif self.gateType == GateType.T:
            return TGate()
        elif self.gateType == GateType.T_dagger:
            return TDaggerGate()
        elif self.gateType == GateType.CZ:
            return CZGate()
        elif self.gateType == GateType.CX:
            return CXGate()
        elif self.gateType == GateType.CY:
            return CYGate()
        elif self.gateType == GateType.CH:
            return CHGate()
        elif self.gateType == GateType.CRz:
            return CRzGate()
        elif self.gateType == GateType.CCX:
            return CCXGate()
        elif self.gateType == GateType.Measure:
            return MeasureGate()
        elif self.gateType == GateType.Swap:
            return SwapGate()
        elif self.gateType == GateType.Perm:
            return PermGate()
        elif self.gateType == GateType.Custom:
            return CustomGate()
        elif self.gateType == GateType.Reset:
            return ResetGate()
        raise Exception("the gate type of the builder is wrong")

    def _inner_complete_gate(self, gate : BasicGate):
        """ private tool function

        filled the initial gate by the parameters set for builder

        Return:
            BasicGate: the gate with parameters set in the builder
        """
        if self.gateType == GateType.Perm:
            gate = gate(self.pargs)
        elif self.gateType == GateType.Custom:
            gate = gate(self.pargs)
        if gate.targets != 0:
            if len(self.targs) == gate.targets:
                gate.targs = copy.deepcopy(self.targs)
            else:
                raise Exception("the number of targs is wrong")

        if gate.controls != 0:
            if len(self.cargs) == gate.controls:
                gate.cargs = copy.deepcopy(self.cargs)
            else:
                raise Exception("the number of cargs is wrong")
        if gate.params != 0 and self.gateType != GateType.Perm:
            if len(self.pargs) == gate.params:
               gate.pargs = copy.deepcopy(self.pargs)
            else:
                raise Exception("the number of pargs is wrong")

        return gate

    @staticmethod
    def apply_gates(gate: BasicGate, circuit: Circuit):
        """ act a gate on some circuit.

        Args:
            gate(BasicGate): the gate which is to be act on the circuit.
            circuit(Circuit): the circuit which the gate acted on.
        """

        qubits = Qureg()
        for control in gate.cargs:
            qubits.append(circuit[control])
        for target in gate.targs:
            qubits.append(circuit[target])
        circuit.add_gate(gate, qubits)

    @staticmethod
    def reflect_gates(gates : list):
        """ build the inverse of a series of gates.

        Args:
            gates(list<BasicGate>): the gate list whose inverse is need to be gotten.

        Return:
            list<BasicGate>: the inverse of the gate list.
        """

        reflect = []
        l_g = len(gates)
        for index in range(l_g - 1, -1, -1):
            reflect.append(gates[index].inverse())
        return reflect

    @staticmethod
    def reflect_apply_gates(gates: list, circuit: Circuit):
        """ act the inverse of a series of gates on some circuit.

        Args:
            gates(list<BasicGate>): the gate list whose inverse is need to be gotten.
            circuit(Circuit): the circuit which the inverse acted on.
        """

        l_g = len(gates)
        for index in range(l_g - 1, -1, -1):
            gate = gates[index].inverse()
            qubits = Qureg()
            for control in gate.cargs:
                qubits.append(circuit[control])
            for target in gate.targs:
                qubits.append(circuit[target])
            circuit.add_gate(gate, qubits)

GateBuilder = GateBuilderModel()

class ExtensionGateType(Enum):
    """ indicate the type of a complex gate

    Every Gate have a attribute named type, which indicate its type.
    """

    QFT = 0
    IQFT = 1
    RZZ = 2
    CU1 = 3
    CU3 = 4
    Fredkin = 5
    CCX = 6
    CRz = 7
    CCRz = 8

class gateModel(object):
    """ the abstract SuperClass of all complex quantum gates

    These quantum gates are generally too complex to act on reality quantum
    hardware directyly. The class is devoted to give some reasonable synthetize
    of the gates so that user can use these gates as basic gates but get a
    series one-qubit and two-qubit gates in final.

    All complex quantum gates described in the framework have
    some common attributes and some common functions
    which defined in this class.

    Note that all subClass must overloaded the function "build_gate", the overloaded
    of the function "__or__" and "__call__" is optional.

    Attributes:
        controls(list<int>): the number of the control bits of the gate
        cargs(list<int>): the list of the index of control bits in the circuit
        carg(int, read only): the first object of cargs

        targets(list<int>): the number of the target bits of the gate
        targs(list<int>): the list of the index of target bits in the circuit
        targ(int, read only): the first object of targs

        params(list): the number of the parameter of the gate
        pargs(list): the list of the parameter
        prag(read only): the first object of pargs

        type(GateType, read only): gate's type described by ExtensionGateType
    """

    # 门对应控制位数
    @property
    def controls(self) -> int:
        return self.__controls

    @controls.setter
    def controls(self, controls: int):
        self.__controls = controls

    # 门对应控制位索引
    @property
    def cargs(self):
        """
        :return:
            返回一个list，表示控制位
        """
        return self.__cargs

    @cargs.setter
    def cargs(self, cargs: list):
        if isinstance(cargs, list):
            self.__cargs = cargs
        else:
            self.__cargs = [cargs]

    # 门对应作用位数
    @property
    def targets(self) -> int:
        return self.__targets

    @targets.setter
    def targets(self, targets: int):
        self.__targets = targets

    # 门对应作用位索引
    @property
    def targs(self):
        """
        :return:
            返回一个list，代表作用位的list
        """
        return self.__targs

    @targs.setter
    def targs(self, targs: list):
        if isinstance(targs, list):
            self.__targs = targs
        else:
            self.__targs = [targs]

    # 辅助数组位个数
    @property
    def params(self) -> int:
        return self.__params

    @params.setter
    def params(self, params: int):
        self.__params = params

    # 辅助数组数组
    @property
    def pargs(self):
        """
        :return:
            返回一个list，代表辅助数组
        """
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

    def __init__(self):
        self.__cargs = []
        self.__targs = []
        self.__pargs = []
        self.__controls = 0
        self.__targets = 0
        self.__params = 0

    @staticmethod
    def qureg_trans(other):
        """ tool function that change tuple/list/Circuit to a Qureg

        For convince, the user can input tuple/list/Circuit/Qureg, but developer
        need only deal with Qureg

        Args:
            other: the item is to be transformed, it can have followed form:
                1) Circuit
                2) Qureg
                3) tuple<qubit, qureg>
                4) list<qubit, qureg>
        Returns:
            Qureg: the qureg transformed into.

        Raises:
            TypeException: the input form is wrong.
        """
        if isinstance(other, tuple):
            other = list(other)
        if isinstance(other, list):
            qureg = Qureg()
            for item in other:
                if isinstance(item, Qubit):
                    qureg.append(item)
                elif isinstance(item, Qureg):
                    qureg.extend(item)
                else:
                    raise TypeException("qubit or tuple<qubit, qureg> or qureg或list<qubit, qureg> or circuit", other)
        elif isinstance(other, Qureg):
            qureg = other
        elif isinstance(other, Circuit):
            qureg = Qureg(other.qubits)
        else:
            raise TypeException("qubit or tuple<qubit> or qureg or circuit", other)
        return qureg

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
            return False

    def __or__(self, other):
        """deal the operator '|'

        Use the syntax "gate | circuit" or "gate | qureg" or "gate | qubit"
        to add the gate into the circuit
        When a one qubit gate act on a qureg or a circuit, it means Adding
        the gate on all the qubit of the qureg or circuit
        Some Examples are like this:

        QFT       | circuit
        IQFT      | circuit([i for i in range(n - 2)])

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) Circuit
                2) qureg
                3) tuple<qubit, qureg>
                4) list<qubit, qureg>
        Raise:
            TypeException: the type of other is wrong
        """

        if isinstance(other, tuple):
            other = list(other)
        if isinstance(other, list):
            qureg = Qureg()
            for item in other:
                if isinstance(item, Qubit):
                    qureg.append(item)
                elif isinstance(item, Qureg):
                    qureg.extend(item)
                else:
                    raise TypeException("qubit或tuple<qubit, qureg>或qureg或list<qubit, qureg>或circuit", other)
        elif isinstance(other, Qureg):
            qureg = other
        elif isinstance(other, Circuit):
            qureg = Qureg(other.qubits)
        else:
            raise TypeException("qubit或tuple<qubit>或qureg或circuit", other)

        gates = self.build_gate(len(qureg))
        if isinstance(gates, Circuit):
            gates = gates.gates
        for gate in gates:
            qubits = []
            for control in gate.cargs:
                qubits.append(qureg[control])
            for target in gate.targs:
                qubits.append(qureg[target])
            qureg.circuit.__add_qureg_gate__(gate, qubits)

    def __call__(self, params):
        """ give parameters for the gate

        give parameters by "()".

        Args:
            params: give parameters for the gate, it can have following form,
                1) int/float/complex
                2) list<int/float/complex>
                3) tuple<int/float/complex>
        Raise:
            TypeException: the type of params is wrong

        Returns:
            gateModel: the gate after filled by parameters
        """
        if self.permit_element(params):
            self.pargs = [params]
        elif isinstance(params, list):
            self.pargs = []
            for element in params:
                if not self.permit_element(element):
                    raise TypeException("int or float or complex", element)
                self.pargs.append(element)
        elif isinstance(params, tuple):
            self.pargs = []
            for element in params:
                if not self.permit_element(element):
                    raise TypeException("int or float or complex", element)
                self.pargs.append(element)
        else:
            raise TypeException("int/float/complex or list<int/float/complex> or tuple<int/float/complex>", params)
        return self

    def build_gate(self, qureg):
        """ the overloaded the build_gate can return two kind of values:
            1)list<BasicGate>: in this way, developer use gateBuilder to generator a series of gates
            2)Circuit: in this way, developer can generator a circuit whose bits number is same as the
                qureg the gate, and apply gates on in. for Example:
                    qureg = self.qureg_trans(qureg)
                    circuit = len(qureg)
                    X | circuit
                    return X
        Args:
            qureg: the gate
        Returns:
            Circuit/list<BasicGate>: synthetize result
        """
        qureg = self.qureg_trans(qureg)
        GateBuilder.setGateType(GateType.X)
        GateBuilder.setTargs(len(qureg) - 1)
        return [GateBuilder.getGate()]

class QFTModel(gateModel):
    """ QFT oracle

    """
    def __or__(self, other):
        """ It can be removed after code refactoring

        """
        qureg = self.qureg_trans(other)
        for i in range(len(other)):
            H | qureg[i]
            for j in range(i + 1, len(other)):
                CRz(2 * np.pi / (1 << j - i + 1)) | (qureg[j], qureg[i])

    def build_gate(self, other):
        gates = []
        for i in range(len(other)):
            GateBuilder.setGateType(GateType.H)
            GateBuilder.setTargs(other[i])
            gates.append(GateBuilder.getGate())

            GateBuilder.setGateType(GateType.CRz)
            for j in range(i + 1, len(other)):
                GateBuilder.setPargs(2 * np.pi / (1 << j - i + 1))
                GateBuilder.setCargs(other[j])
                GateBuilder.setTargs(other[i])
                gates.append(GateBuilder.getGate())
        return gates

QFT = QFTModel()

class IQFTModel(gateModel):
    """ IQFT gate

    """
    def __or__(self, other):
        """ It can be removed after code refactoring

        """
        qureg = self.qureg_trans(other)
        for i in range(len(other) - 1, -1, -1):
            for j in range(len(other) - 1, i, -1):
                CRz(-2 * np.pi / (1 << j - i + 1)) | (qureg[j], qureg[i])
            H | qureg[i]

    def build_gate(self, other):
        gates = []
        for i in range(len(other) - 1, -1, -1):
            GateBuilder.setGateType(GateType.CRz)
            for j in range(len(other) - 1, i, -1):
                GateBuilder.setPargs(-2 * np.pi / (1 << j - i + 1))
                GateBuilder.setCargs(other[j])
                GateBuilder.setTargs(other[i])
                gates.append(GateBuilder.getGate())
            GateBuilder.setGateType(GateType.H)
            GateBuilder.setTargs(other[i])
            gates.append(GateBuilder.getGate())
        return gates

IQFT = IQFTModel()

class RZZModel(gateModel):
    """ RZZ gate

    """

    def __or__(self, other):
        """ It can be removed after code refactoring

        """
        qureg = self.qureg_trans(other)
        CX | (qureg[0], qureg[1])
        U1(self.parg) | qureg[1]
        CX | (qureg[0], qureg[1])

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.U1)
        GateBuilder.setPargs(self.parg)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

RZZ = RZZModel()

class CU1Gate(gateModel):
    """ Controlled-U1 gate

    """
    def __or__(self, other):
        """ It can be removed after code refactoring

        """
        qureg = self.qureg_trans(other)
        U1(self.parg / 2) | qureg[0]
        CX | (qureg[0], qureg[1])
        U1(-self.parg / 2) | qureg[1]
        CX | (qureg[0], qureg[1])
        U1(self.parg / 2) | qureg[1]

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.U1)
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.U1)
        GateBuilder.setPargs(-self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.U1)
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

CU1 = CU1Gate()

class CRz_DecomposeModel(gateModel):
    """ Controlled-Rz gate

    """
    def __or__(self, other):
        """ It can be removed after code refactoring

        """
        qureg = self.qureg_trans(other)
        Rz(self.parg / 2) | qureg[0]
        CX | (qureg[0], qureg[1])
        Rz(-self.parg / 2) | qureg[1]
        CX | (qureg[0], qureg[1])
        Rz(self.parg / 2) | qureg[1]

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.Rz)
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.Rz)
        GateBuilder.setPargs(-self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.Rz)
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

CRz_Decompose = CRz_DecomposeModel()

class CU3Gate(gateModel):
    """ controlled-U3 gate

    """
    def __or__(self, other):
        """ It can be removed after code refactoring

        """
        qureg = self.qureg_trans(other)
        U1((self.pargs[1] + self.pargs[2]) / 2) | qureg[0]
        U1(self.pargs[2] - self.pargs[1]) | (qureg[1])
        CX | (qureg[0], qureg[1])
        U3([-self.pargs[0] / 2, 0, -(self.pargs[0] + self.pargs[1]) / 2]) | qureg[1]
        CX | (qureg[0], qureg[1])
        U3([self.pargs[0] / 2, self.pargs[1], 0]) | qureg[1]

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.U1)
        GateBuilder.setPargs((self.pargs[1] + self.pargs[2]) / 2)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setPargs(self.pargs[2] + self.pargs[1])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.U3)
        GateBuilder.setPargs([-self.pargs[0] / 2, 0, -(self.pargs[0] + self.pargs[1]) / 2])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.U3)
        GateBuilder.setPargs([self.pargs[0] / 2, self.pargs[1], 0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

CU3 = CU3Gate()

class CCRzModel(gateModel):
    """ controlled-Rz gate with two control bits

    """
    def __or__(self, other):
        """ It can be removed after code refactoring

        """
        qureg = self.qureg_trans(other)
        CRz_Decompose(self.parg / 2)  | (qureg[1], qureg[2])
        CX                            | (qureg[0], qureg[1])
        CRz_Decompose(-self.parg / 2) | (qureg[1], qureg[2])
        CX                            | (qureg[0], qureg[1])
        CRz_Decompose(self.parg / 2)  | (qureg[0], qureg[2])

    def build_gate(self, other):
        qureg = Circuit(3)
        CRz_Decompose(self.parg / 2) | (qureg[1], qureg[2])
        CX | (qureg[0], qureg[1])
        CRz_Decompose(-self.parg / 2) | (qureg[1], qureg[2])
        CX | (qureg[0], qureg[1])
        CRz_Decompose(self.parg / 2) | (qureg[0], qureg[2])

        return qureg

CCRz = CCRzModel()

class FredkinModel(gateModel):
    def __or__(self, other):
        """ It can be removed after code refactoring

        """
        qureg = self.qureg_trans(other)
        CX            | (qureg[2], qureg[1])
        CCX_Decompose | (qureg[0], qureg[1], qureg[2])
        CX            | (qureg[2], qureg[1])

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[2])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        gates.extend(CCX_Decompose.build_gate(other))

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[2])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

Fredkin = FredkinModel()

class CCX_DecomposeModel(gateModel):
    def __or__(self, other):
        """ It can be removed after code refactoring

        """
        qureg = self.qureg_trans(other)
        H           | qureg[2]
        CX          | (qureg[1], qureg[2])
        T_dagger    | qureg[2]
        CX          | (qureg[0], qureg[2])
        T           | qureg[2]
        CX          | (qureg[1], qureg[2])
        T_dagger    | qureg[2]
        CX          | (qureg[0], qureg[2])
        T           | qureg[1]
        T           | qureg[2]
        H           | qureg[2]
        CX          | (qureg[0], qureg[1])
        T           | qureg[0]
        T_dagger    | qureg[1]
        CX          | (qureg[0], qureg[1])

    def build_gate(self, other):
        gates = []

        GateBuilder.setGateType(GateType.H)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[1])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T_dagger)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[1])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T_dagger)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.H)
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.T_dagger)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

CCX_Decompose = CCX_DecomposeModel()

class ExtensionGateBuilderModel(object):
    """ A model that help users get gate without circuit

    The model is designed to help users get some gates independent of the circuit
    Because there is no clear API to setting a gate's control bit indexes and
    target bit indexes without circuit or qureg.

    Users should set the gateType of the ExtensionGateBuilder, than set necessary parameters
    (Targs, Cargs, Pargs). After that, user can get a gate from ExtensionGateBuilder.

    """

    def __init__(self):
        self.gateType = GateType.Error
        self.pargs = []
        self.targs = []
        self.cargs = []

    def setGateType(self, type):
        self.gateType = type

    def setPargs(self, pargs):
        """ pass the parameters of the gate

        if the gate don't need the parameters, needn't to call this function.

        Args:
            pargs(list/int/float/complex): the parameters filled in the gate
        """

        if isinstance(pargs, list):
            self.pargs = pargs
        else:
            self.pargs = [pargs]

    def setTargs(self, targs):
        """ pass the target bits' indexes of the gate

        The targets should be passed.

        Args:
            targs(list/int/float/complex): the target bits' indexes the gate act on.
        """
        if isinstance(targs, list):
            self.targs = targs
        else:
            self.targs = [targs]

    def getTargsNumber(self):
        """ get the number of targs of the gate

        once the gateType is set, the function is valid.

        Return:
            int: the number of targs
        """

        gate = self._inner_generate_gate()
        return gate.targets

    def getParamsNumber(self):
        """ get the number of pargs of the gate

        once the gateType is set, the function is valid.

        Return:
            int: the number of pargs
        """
        gate = self._inner_generate_gate()
        return gate.params

    def getGate(self):
        """ get the gate

        once the parameters are set, the function is valid.

        Return:
            gateModel: the gate with parameters set in the builder
        """
        return self._inner_generate_gate()

    def _inner_generate_gate(self):
        """ private tool function

        get an initial gate by the gateType set for builder

        Return:
            BasicGate: the initial gate
        """
        if self.gateType == ExtensionGateType.QFT:
            return QFT.build_gate(self.targs)
        elif self.gateType == ExtensionGateType.IQFT:
            return IQFT.build_gate(self.targs)
        elif self.gateType == ExtensionGateType.RZZ:
            return RZZ(self.pargs).build_gate(self.targs)
        elif self.gateType == ExtensionGateType.CU1:
            return CU1(self.pargs).build_gate(self.targs)
        elif self.gateType == ExtensionGateType.CU3:
            return CU3(self.pargs).build_gate(self.targs)
        elif self.gateType == ExtensionGateType.Fredkin:
            return Fredkin.build_gate(self.targs)
        elif self.gateType == ExtensionGateType.CCX:
            return CCX_Decompose.build_gate(self.targs)
        elif self.gateType == ExtensionGateType.CRz:
            return CRz_Decompose(self.pargs).build_gate(self.targs)
        elif self.gateType == ExtensionGateType.CCRz:
            return CCRz(self.pargs).build_gate(self.targs)

        raise Exception("the gate type of the builder is wrong")

ExtensionGateBuilder = ExtensionGateBuilderModel()

class GateDigitException(Exception):
    def __init__(self, controls, targets, indeed):
        """
        Args:
            controls(int): the number of controls
            targets(int) : the number of targets
            indeed(int)  : indeed the number of indexed passed in
        """
        Exception.__init__(self, f"the number of control bits indexes is {controls} ,\
                                 the number of target bits indexed is {targets}, \
                                 so {controls + targets} parameters should be passed in, \
                                 infact {indeed} parameters are passed in.")
