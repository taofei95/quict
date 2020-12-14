#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39
# @Author  : Han Yu
# @File    : _synthesis.py

import numpy as np

from QuICT.core.exception import TypeException
from QuICT.core import Circuit, GateBuilder, Qubit, Qureg

class Synthesis(object):
    """ synthesis some oracle into BasicGate

    In general, these BasicGates are one-qubit gates and two-qubit gates.

    in _gate.py, we define a class named gateModel, which is similar with this class.
    The difference of them is that gateModel describes simple and widely accepted to a certain extent.
    And this class describes harder synthesis method, some of which may have a lot room to improve.

    When users use the synthesis method, it should be similar with class gateModel. So the attributes and
    API is similar with class gateModel.

    Note that all subClass must overloaded the function "build_gate".
    If the algorithm have any parameters to be input, "__call__" should be overloaded.
    The overloaded of the function "__or__" and "__xor__" is optional.

    Attributes:
        targets(int): the number of the target bits of the gate
                      Note it will be assigned after calling the function "__or__"
        targs(list<int>): the list of the index of target bits in the circuit
        targ(int, read only): the first object of targs

        pargs(list): the list of the parameter
        prag(read only): the first object of pargs

    """

    def __init__(self):
        self.__pargs = []
        self.__targets = 0

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
    def targets(self):
        return self.__targets

    @targets.setter
    def targets(self, targets):
        self.__targets = targets

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

    def __or__(self, other):
        """deal the operator '|'

        Use the syntax "oracle | circuit" or "oracle | qureg" or "oracle | qubit"
        to add the oracle into the circuit
        When a one qubit gate act on a qureg or a circuit, it means Adding
        the gate on all the qubit of the qureg or circuit

        targets/targs/targ will be assigned when calling this function

        Some Examples are like this:

        MCT_one_aux                 | circuit
        MCT_Linear_Simulation       | circuit([i for i in range(n - 2)])

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
                    raise TypeException("qubit or tuple<qubit, qureg> or qureg or list<qubit, qureg> or circuit", other)
        elif isinstance(other, Qureg):
            qureg = other
        elif isinstance(other, Circuit):
            qureg = Qureg(other.qubits)
        else:
            raise TypeException("qubit or tuple<qubit> or qureg or circuit", other)

        self.targets = len(qureg)

        gates = self.build_gate()
        if isinstance(gates, Circuit):
            gates = gates.gates
        for gate in gates:
            qubits = []
            for control in gate.cargs:
                qubits.append(qureg[control])
            for target in gate.targs:
                qubits.append(qureg[target])
            qureg.circuit.append(gate, qubits)

    def __xor__(self, other):
        """deal the operator '^'

        Use the syntax "oracle ^ circuit" or "oracle ^ qureg" or "oracle ^ qubit"
        to add the inverse of the oracle into the circuit
        When a one qubit gate act on a qureg or a circuit, it means Adding
        the gate on all the qubit of the qureg or circuit
        Some Examples are like this:

        MCT_one_aux                 | circuit
        MCT_Linear_Simulation       | circuit([i for i in range(n - 2)])

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
                    raise TypeException("qubit or tuple<qubit, qureg> or qureg or list<qubit, qureg> or circuit", other)
        elif isinstance(other, Qureg):
            qureg = other
        elif isinstance(other, Circuit):
            qureg = Qureg(other.qubits)
        else:
            raise TypeException("qubit or tuple<qubit> or qureg or circuit", other)

        gates = self.build_gate()
        if isinstance(gates, Circuit):
            gates = gates.gates
        gates = GateBuilder.reflect_gates(gates)
        for gate in gates:
            qubits = []
            for control in gate.cargs:
                qubits.append(qureg[control])
            for target in gate.targs:
                qubits.append(qureg[target])
            qureg.circuit.append(gate, qubits)

    def __call__(self, *pargs):
        raise Exception('"__call__" function must be overloaded')

    def build_gate(self):
        raise Exception('"build_gate" function must be overloaded')

