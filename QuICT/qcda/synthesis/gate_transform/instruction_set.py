#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:31 下午
# @Author  : Han Yu
# @File    : instruction_set.py

from QuICT.core import BasicGate, Circuit, GATE_ID
from .transform_rule import TransformRule

class InstructionSet(object):
    """ InstructionSet describes a set of gates(expectly to be universal set)

    Instruction Set contains gates and some rules, which can be assigned by user.

    Attributes:
        two_qubit_gate(int): the index of the two_qubit_gate
        one_qubit_gates(list<int>): the indexes of the one_qubit_gate
        SU4_rule(TransformRule): rules to transform SU(4) into instruction set
        SU2_rule(TransformRule): rules to transform SU(2) into instruction set
        rule_map(dictionary): A two-dimensional map from source gate and target gate to transform rule

    """

    @property
    def two_qubit_gate(self):
        return self.__two_qubit_gate

    @two_qubit_gate.setter
    def two_qubit_gate(self, two_qubit_gate):
        """ set two_qubit_gate

        the basicGate class is transformed to gate id

        Args:
            two_qubit_gate(int/BasicGate):
        """
        if isinstance(two_qubit_gate, BasicGate):
            two_qubit_gate = GATE_ID[two_qubit_gate.type()]
        self.__two_qubit_gate = two_qubit_gate

    @property
    def one_qubit_gates(self):
        return self.__one_qubit_gates

    @one_qubit_gates.setter
    def one_qubit_gates(self, one_qubit_gates):
        """ set one_qubit_gates

        the basicGate class is transformed to gate id

        Args:
            two_qubit_gate(list<int/BasicGate>):
        """
        one_qubit_gates_list = []
        for element in one_qubit_gates:
            if isinstance(element, BasicGate):
                one_qubit_gates_list.append(element.type())
            else:
                one_qubit_gates_list.append(element)
        self.__one_qubit_gates = one_qubit_gates

    @property
    def SU4_rule(self) -> TransformRule:
        return self.__SU4_rule

    @property
    def SU2_rule(self) -> TransformRule:
        return self.__SU2_rule

    @property
    def rule_map(self):
        return self.__rule_map

    @rule_map.setter
    def rule_map(self, rule_map):
        self.__rule_map = rule_map

    def __init__(self, two_qubit_gate = 0, one_qubit_gates = None):
        self.two_qubit_gate = two_qubit_gate
        if one_qubit_gates is None:
            one_qubit_gates = []
        self.one_qubit_gates = one_qubit_gates
        self.__SU4_rule = None
        self.__SU2_rule = None
        self.__rule_map = {}
        self.select_default_rule(self.two_qubit_gate, self.one_qubit_gates)

    def select_default_rule(self, two_qubit_gate, one_qubit_gates):
        """ when setting the gates, select the default rules.

        Args:
            two_qubit_gate(int): the id of 2-qubit gate
            one_qubit_gates(list<int>): the ids of 1-qubit gates
        """
        pass

    def select_transform_rule(self, source, target) ->  TransformRule:
        """ choose a rule which transforms source gate into target gate

        Args:
            source(int): the id of source gate
            target(int): the id of target gate

        Returns:
            TransformRule: the transform rules
        """
        pass

    def register_SU2_rule(self, function):
        """ register SU(2) decompostion rule

        Args:
            function(TransformRule): decompostion rule
        """
        pass

    def register_SU4_rule(self, function):
        """ register SU(4) decompostion rule

        Args:
            function(TransformRule): decompostion rule
        """
        pass

    def register_rule_map(self, function):
        """ register rule which transforms from source gate into target gate

        Args:
            function(TransformRule): the transform rule
        """
        pass

    def batch_register_rule_map(self, functions):
        """ batch register rules which transforms from source gate into target gate

        Args:
            functions(list<TransformRule>): the transform rules
        """
        pass
