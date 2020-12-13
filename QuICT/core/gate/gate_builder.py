#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/11 10:54 上午
# @Author  : Han Yu
# @File    : _gateBuilder.py

from .gate import *

class GateBuilderModel(object):
    """ A model that help users get gate without circuit

    The model is designed to help users get some gates independent of the circuit
    Because there is no clear API to setting a gate's control bit indexes and
    target bit indexes without circuit or qureg.

    Users should set the gateType of the GateBuilder, than set necessary parameters
    (Targs, Cargs, Pargs). After that, user can get a gate from GateBuilder.

    """

    def __init__(self):
        self.gateType = GATE_ID["Error"]
        self.pargs = []
        self.cargs = []
        self.targs = []

    def setGateType(self, type):
        """ pass the gateType for the builder

        Args:
            type(int): the type passed in
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
        if self.gateType == GATE_ID["H"]:
            return HGate()
        elif self.gateType == GATE_ID["S"]:
            return SGate()
        elif self.gateType == GATE_ID["S_dagger"]:
            return SDaggerGate()
        elif self.gateType == GATE_ID["X"]:
            return XGate()
        elif self.gateType == GATE_ID["Y"]:
            return YGate()
        elif self.gateType == GATE_ID["Z"]:
            return ZGate()
        elif self.gateType == GATE_ID["ID"]:
            return IDGate()
        elif self.gateType == GATE_ID["U1"]:
            return U1Gate()
        elif self.gateType == GATE_ID["U2"]:
            return U2Gate()
        elif self.gateType == GATE_ID["U3"]:
            return U3Gate()
        elif self.gateType == GATE_ID["Rx"]:
            return RxGate()
        elif self.gateType == GATE_ID["Ry"]:
            return RyGate()
        elif self.gateType == GATE_ID["Rz"]:
            return RzGate()
        elif self.gateType == GATE_ID["T"]:
            return TGate()
        elif self.gateType == GATE_ID["T_dagger"]:
            return TDaggerGate()
        elif self.gateType == GATE_ID["CZ"]:
            return CZGate()
        elif self.gateType == GATE_ID["CX"]:
            return CXGate()
        elif self.gateType == GATE_ID["CY"]:
            return CYGate()
        elif self.gateType == GATE_ID["CH"]:
            return CHGate()
        elif self.gateType == GATE_ID["CRz"]:
            return CRzGate()
        elif self.gateType == GATE_ID["CCX"]:
            return CCXGate()
        elif self.gateType == GATE_ID["Measure"]:
            return MeasureGate()
        elif self.gateType == GATE_ID["Swap"]:
            return SwapGate()
        elif self.gateType == GATE_ID["Perm"]:
            return PermGate()
        elif self.gateType == GATE_ID["Custom"]:
            return CustomGate()
        elif self.gateType == GATE_ID["Reset"]:
            return ResetGate()
        raise Exception("the gate type of the builder is wrong")

    def _inner_complete_gate(self, gate: BasicGate):
        """ private tool function

        filled the initial gate by the parameters set for builder

        Return:
            BasicGate: the gate with parameters set in the builder
        """
        if self.gateType == GATE_ID["Perm"]:
            gate = gate(self.pargs)
        elif self.gateType == GATE_ID["Custom"]:
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
        if gate.params != 0 and self.gateType != GATE_ID['Perm']:
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
        circuit.append(gate, qubits)

    @staticmethod
    def reflect_gates(gates: list):
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
            circuit.append(gate, qubits)

GateBuilder = GateBuilderModel()
