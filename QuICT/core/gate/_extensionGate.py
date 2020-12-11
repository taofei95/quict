#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/11 10:55 上午
# @Author  : Han Yu
# @File    : _extensionGate.py

from ._gate import *
from ._gateBuilder import GateBuilder

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
        controls(int): the number of the control bits of the gate
        cargs(list<int>): the list of the index of control bits in the circuit
        carg(int, read only): the first object of cargs

        targets(int): the number of the target bits of the gate
        targs(list<int>): the list of the index of target bits in the circuit
        targ(int, read only): the first object of targs

        params(list): the number of the parameter of the gate
        pargs(list): the list of the parameter
        prag(read only): the first object of pargs

        type(GateType, read only): gate's type described by ExtensionGateType
    """

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
            tp = type(element)
            if tp == np.int64 or tp == np.float or tp == np.complex128:
                return True
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
            qureg.circuit.append(gate, qubits)

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