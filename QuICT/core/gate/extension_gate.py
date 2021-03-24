#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/11 10:55
# @Author  : Han Yu
# @File    : _extensionGate.py

from .gate import *
from .gate_set import *
from .gate_builder import GateBuilder

def _add_alias(alias, standard_name):
    if alias is not None:
        global EXTENSION_GATE_ID
        if isinstance(alias, str):
            EXTENSION_GATE_ID[alias] = EXTENSION_GATE_ID[standard_name]
        else:
            for nm in alias:
                if nm in EXTENSION_GATE_ID:
                    continue
                EXTENSION_GATE_ID[nm] = EXTENSION_GATE_ID[standard_name]

EXTENSION_GATE_REGISTER = {-1: "Error"}
""" Get standard gate name by gate id.
"""

EXTENSION_GATE_ID = {"Error": -1}
""" Get gate id by gate name. You may use any one of the aliases of this gate.
"""

EXTENSION_GATE_ID_CNT = 0
""" Gate number counter.
"""

def extension_gate_implementation(cls):
    global EXTENSION_GATE_REGISTER
    global EXTENSION_GATE_ID
    global EXTENSION_GATE_ID_CNT

    EXTENSION_GATE_REGISTER[EXTENSION_GATE_ID_CNT] = cls
    EXTENSION_GATE_ID[cls.__name__] = EXTENSION_GATE_ID_CNT
    EXTENSION_GATE_ID_CNT += 1

    @functools.wraps(cls)
    def gate_variation(*args, **kwargs):
        return cls(*args, **kwargs)

    return gate_variation

class gateModel(object):
    """ the abstract SuperClass of all complex quantum gates

    These quantum gates are generally too complex to act on reality quantum
    hardware directly. The class is devoted to give some reasonable synthetize
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

    def __init_subclass__(cls, **kwargs):
        return extension_gate_implementation(cls)

    def __init__(self, alias=None):
        self.__cargs = []
        self.__targs = []
        self.__pargs = []
        self.__controls = 0
        self.__targets = 0
        self.__params = 0
        _add_alias(alias=alias, standard_name=self.__class__.__name__)

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
                    raise TypeException("qubit or tuple<qubit, qureg> or qureg or list<qubit, qureg> or circuit", other)
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

    def __or__(self, targets):
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

        try:
            qureg = Qureg(targets)
            circuit = qureg.circuit
            indexes = [i for i in range(len(qureg))]
            gates = self.build_gate(indexes)
            if isinstance(gates, Circuit):
                gates = gates.gates
            for gate in gates:
                qubits = []
                for control in gate.cargs:
                    qubits.append(qureg[control])
                for target in gate.targs:
                    qubits.append(qureg[target])
                circuit.append(gate, qubits)
        except Exception:
            raise TypeException("qubit or tuple<qubit, qureg> or qureg or list<qubit, qureg> or circuit", targets)

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

    def copy(self):
        """ copy the gate

        """
        name = str(self.__class__.__name__)
        gate = globals()[name]()
        gate.pargs = copy.deepcopy(self.pargs)
        gate.targs = copy.deepcopy(self.targs)
        gate.cargs = copy.deepcopy(self.cargs)
        gate.targets = self.targets
        gate.controls = self.controls
        gate.params = self.params
        return gate

    def build_gate(self, qureg = None):
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
        GateBuilder.setGateType(GATE_ID["X"])
        GateBuilder.setTargs(len(qureg) - 1)
        return [GateBuilder.getGate()]


class QFTModel(gateModel):
    """ QFT oracle

    """

    def build_gate(self, other = None):
        gates = []
        for i in range(len(other)):
            GateBuilder.setGateType(GATE_ID["H"])
            GateBuilder.setTargs(other[i])
            gates.append(GateBuilder.getGate())

            GateBuilder.setGateType(GATE_ID["CRz"])
            for j in range(i + 1, len(other)):
                GateBuilder.setPargs(2 * np.pi / (1 << j - i + 1))
                GateBuilder.setCargs(other[j])
                GateBuilder.setTargs(other[i])
                gates.append(GateBuilder.getGate())
        return gates


QFT = QFTModel(['QFT', 'qft'])


class IQFTModel(gateModel):
    """ IQFT gate

    """

    def build_gate(self, other = None):
        gates = []
        for i in range(len(other) - 1, -1, -1):
            GateBuilder.setGateType(GATE_ID["CRz"])
            for j in range(len(other) - 1, i, -1):
                GateBuilder.setPargs(-2 * np.pi / (1 << j - i + 1))
                GateBuilder.setCargs(other[j])
                GateBuilder.setTargs(other[i])
                gates.append(GateBuilder.getGate())
            GateBuilder.setGateType(GATE_ID["H"])
            GateBuilder.setTargs(other[i])
            gates.append(GateBuilder.getGate())
        return gates


IQFT = IQFTModel(['IQFT', 'iqft'])


class RZZModel(gateModel):
    """ RZZ gate

    """

    def build_gate(self, other = None):
        gates = []

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["Rz"])
        GateBuilder.setPargs(self.parg)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates


RZZ = RZZModel(['RZZ', 'Rzz', 'rzz'])


class CU1Gate(gateModel):
    """ Controlled-U1 gate

    """

    def build_gate(self, other = None):
        if other is None:
            other = self.targs
        gates = []

        GateBuilder.setGateType(GATE_ID["U1"])
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["U1"])
        GateBuilder.setPargs(-self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["U1"])
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates


CU1 = CU1Gate(["CU1", "cu1"])


class CRz_DecomposeModel(gateModel):
    """ Controlled-Rz gate

    """

    def build_gate(self, other = None):
        if other is None:
            other = self.targs
        gates = []

        GateBuilder.setGateType(GATE_ID["Rz"])
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["Rz"])
        GateBuilder.setPargs(-self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["Rz"])
        GateBuilder.setPargs(self.parg / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates


CRz_Decompose = CRz_DecomposeModel(["CRz", "crz"])


class CU3Gate(gateModel):
    """ controlled-U3 gate

    """

    def build_gate(self, other = None):
        if other is None:
            other = self.targs
        gates = []

        GateBuilder.setGateType(GATE_ID["U1"])
        GateBuilder.setPargs((self.pargs[2] + self.pargs[1]) / 2)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setPargs((self.pargs[2] - self.pargs[1]) / 2)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["U3"])
        GateBuilder.setPargs([-self.pargs[0] / 2, 0, -(self.pargs[1] + self.pargs[2]) / 2])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["U3"])
        GateBuilder.setPargs([self.pargs[0] / 2, self.pargs[1], 0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates


CU3 = CU3Gate(["CU3"])


class CCRzModel(gateModel):
    """ controlled-Rz gate with two control bits

    """

    def build_gate(self, other = None):
        if other is None:
            other = self.targs
        qureg = Circuit(3)
        CRz_Decompose(self.parg / 2) | (qureg[1], qureg[2])
        CX | (qureg[0], qureg[1])
        CRz_Decompose(-self.parg / 2) | (qureg[1], qureg[2])
        CX | (qureg[0], qureg[1])
        CRz_Decompose(self.parg / 2) | (qureg[0], qureg[2])

        return qureg


CCRz = CCRzModel(["CCRz"])


class FredkinModel(gateModel):

    def build_gate(self, other = None):
        if other is None:
            other = self.targs
        gates = []

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[2])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        gates.extend(CCX_Decompose.build_gate(other))

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[2])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates


Fredkin = FredkinModel(["Fredkin", "cswap"])


class CCX_DecomposeModel(gateModel):

    def build_gate(self, other = None):
        if other is None:
            other = self.targs
        gates = []

        GateBuilder.setGateType(GATE_ID["H"])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[1])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["T_dagger"])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["T"])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[1])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["T_dagger"])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["T"])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["T"])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["H"])
        GateBuilder.setTargs(other[2])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["T"])
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["T_dagger"])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GATE_ID["CX"])
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        return gates

CCX_Decompose = CCX_DecomposeModel(["CCX", "Toffoli", "toffoli"])
