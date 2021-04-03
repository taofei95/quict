#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:31 下午
# @Author  : Han Yu
# @File    : instruction_set.py

from QuICT.core import BasicGate, Circuit, GATE_ID, GATE_REGISTER
from .transform_rule import TransformRule
from .transform_rule.two_qubit_gate_rules import *

def _capitalize_name_of_gate(class_name):
    """ According to the class_name, generator the capitalized name

    Args:
        class_name(string): class name of the gate

    Returns:
        string: the capitalized name
    """
    index = class_name.find("Gate")
    if index == -1:
        raise Exception("the parameter is not legal.")
    class_name = class_name[:index]
    return class_name.capitalize()

def _generate_default_rule(source, target):
    """ According to the source gate and target gate(2-qubit), choose a appropriate default rule

    Args:
        source(BasicGate/int): source gate (id)
        target(BasicGate/int): target gate (id)

    Returns:
        TransformRule: a valid rule.
    """
    if isinstance(source, BasicGate):
        source = source.type()
    if isinstance(target, BasicGate):
        target = target.type()
    source = _capitalize_name_of_gate(GATE_REGISTER[source].__class__.__name__)
    target = _capitalize_name_of_gate(GATE_REGISTER[target].__class__.__name__)
    rule_name = f"{source}2{target}Rule"
    return eval(rule_name)

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
            two_qubit_gate = two_qubit_gate.type()
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

    def select_transform_rule(self, source) ->  TransformRule:
        """ choose a rule which transforms source gate into target gate(2-qubit)

        Args:
            source(BasicGate/int): the id of source gate

        Returns:
            TransformRule: the transform rules
        """
        if isinstance(source, BasicGate):
            source = source.type()
        if source in self.rule_map:
            return self.rule_map[source]
        rule = _generate_default_rule(source, self.two_qubit_gate)
        self.rule_map[source] = rule
        return rule

    def register_SU2_rule(self, transformRule):
        """ register SU(2) decompostion rule

        Args:
            transformRule(TransformRule): decompostion rule
        """
        self.__SU2_rule = transformRule

    def register_SU4_rule(self, transformRule):
        """ register SU(4) decompostion rule

        Args:
            transformRule(TransformRule): decompostion rule
        """
        self.__SU4_rule = transformRule

    def register_rule_map(self, transformRule):
        """ register rule which transforms from source gate into target gate

        Args:
            transformRule(TransformRule): the transform rule
        """
        if transformRule.target != self.two_qubit_gate:
            raise Exception("the target is not in the target")
        self.rule_map[transformRule.source] = transformRule

    def batch_register_rule_map(self, transformRules):
        """ batch register rules which transforms from source gate into target gate

        Args:
            transformRules(list<TransformRule>): the transform rules
        """
        for transformRule in transformRules:
            self.register_rule_map(transformRule)
