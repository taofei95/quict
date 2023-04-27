#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:31 下午
# @Author  : Han Yu
# @File    : instruction_set.py
from typing import Union, List, Dict
from types import FunctionType

from QuICT.core.gate import GateType


class InstructionSet(object):
    """ InstructionSet describes a set of gates(expectly to be universal set)

    Instruction Set contains gates and some rules, which can be assigned by user.

    Attributes:
        two_qubit_gate (GateType): the type of the two_qubit_gate
        one_qubit_gates (list<GateType>): the types of the one_qubit_gate
        one_qubit_gates_fidelity (Union[float, Dict, List], optional): The fidelity for single qubit quantum gate.
            Defaults to None.
        one_qubit_rule (Union[str, callable], optional): rules to transform SU(2) into instruction set
    """
    @property
    def size(self):
        return len(self.one_qubit_gates) + 1

    @property
    def gates(self) -> list:
        """ Return the list of GateType in current Instruction Set. """
        return self.one_qubit_gates + [self.two_qubit_gate]

    # Two-qubit gate and two-qubit rules
    @property
    def two_qubit_gate(self):
        return self.__two_qubit_gate

    @two_qubit_gate.setter
    def two_qubit_gate(self, two_qubit_gate):
        """ set two_qubit_gate

        Args:
            two_qubit_gate(GateType): two-qubit gate in the InstructionSet
        """
        assert isinstance(two_qubit_gate, GateType), TypeError('two_qubit_gate should be a GateType')
        self.__two_qubit_gate = two_qubit_gate

    @property
    def two_qubit_rule_map(self):
        return self.__two_qubit_rule_map

    @two_qubit_rule_map.setter
    def two_qubit_rule_map(self, two_qubit_rule_map):
        self.__two_qubit_rule_map = two_qubit_rule_map

    # One-qubit gates and one-qubit rule
    @property
    def one_qubit_gates(self):
        return self.__one_qubit_gates

    @one_qubit_gates.setter
    def one_qubit_gates(self, one_qubit_gates):
        """ set one_qubit_gates

        Args:
            one_qubit_gates(list<GateType>): one-qubit gates in the InstructionSet
        """
        assert isinstance(one_qubit_gates, list), TypeError('one_qubit_gates should be a list')
        for one_qubit_gate in one_qubit_gates:
            assert isinstance(one_qubit_gate, GateType), TypeError('each one_qubit_gate should be a GateType')
        self.__one_qubit_gates = one_qubit_gates

    @property
    def one_qubit_rule(self):
        """ the rule of decompose 2*2 unitary into target gates

        If not assigned by the register_one_qubit_rule method, some pre-implemented method would be chosen
        corresponding to the one_qubit_gates. An Exception will be raised when no method is chosen.

        Returns:
            callable: the corresponding rule
        """
        if self.__one_qubit_rule:
            return self.__one_qubit_rule
        if set((GateType.rz, GateType.ry)).issubset(set(self.one_qubit_gates)):
            return "zyz_rule"
        if set((GateType.rz, GateType.rx)).issubset(set(self.one_qubit_gates)):
            return "zxz_rule"
        if set((GateType.rx, GateType.ry)).issubset(set(self.one_qubit_gates)):
            return "xyx_rule"
        if set((GateType.h, GateType.rz)).issubset(set(self.one_qubit_gates)):
            return "hrz_rule"
        if set((GateType.rz, GateType.sx, GateType.x)).issubset(set(self.one_qubit_gates)):
            return "ibmq_rule"
        if set((GateType.u3,)).issubset(set(self.one_qubit_gates)):
            return "u3_rule"
        raise Exception("please register the SU2 decomposition rule.")

    @property
    def one_qubit_fidelity(self):
        return self.__one_qubit_gates_fidelity

    def __init__(
        self,
        two_qubit_gate: GateType,
        one_qubit_gates: List[GateType],
        one_qubit_gates_fidelity: Union[float, List, Dict] = None,
        one_qubit_rule: Union[str, callable] = None
    ):
        self.two_qubit_gate = two_qubit_gate
        self.one_qubit_gates = one_qubit_gates
        self.__one_qubit_gates_fidelity = None
        if one_qubit_gates_fidelity is not None:
            self.register_one_qubit_fidelity(one_qubit_gates_fidelity)

        self.__one_qubit_rule = None
        if one_qubit_rule is not None:
            self.register_one_qubit_rule(one_qubit_rule)

        self.__two_qubit_rule_map = {}

    def select_transform_rule(self, source):
        """ choose a rule which transforms source gate into target gate(2-qubit)

        Args:
            source(GateType): the type of source gate

        Returns:
            callable: the transform rules
        """
        assert isinstance(source, GateType)
        if source in self.two_qubit_rule_map.keys():
            return self.two_qubit_rule_map[source]

        rule = f"{source.name}2{self.two_qubit_gate.name}_rule"
        self.two_qubit_rule_map[source] = rule
        return rule

    def register_one_qubit_fidelity(self, gates_fidelity: Union[float, List, Dict]):
        if isinstance(gates_fidelity, float):
            gates_fidelity = [gates_fidelity] * len(self.one_qubit_gates)
        elif isinstance(gates_fidelity, list):
            assert len(gates_fidelity) == len(self.one_qubit_fidelity)
        elif isinstance(gates_fidelity, dict):
            assert len(gates_fidelity.keys()) == len(self.one_qubit_fidelity)
            for gate_type, fidelity in gates_fidelity.items():
                assert gate_type in self.one_qubit_gates, ValueError(f"Unknown Single-Qubit Gate {gate_type}.")
                assert fidelity >= 0 and fidelity <= 1, \
                    ValueError(f"Wrong Fidelity {fidelity}, it should between 0 and 1.")
        else:
            raise TypeError(f"Unsupport Single-Qubit Gates' Fidelity, {type(gates_fidelity)}.")

        if isinstance(gates_fidelity, list):
            self.__one_qubit_gates_fidelity = {}
            for idx, fidelity in enumerate(gates_fidelity):
                assert fidelity >= 0 and fidelity <= 1, ValueError(
                    f"Wrong Fidelity {fidelity}, it should between 0 and 1."
                )
                self.__one_qubit_gates_fidelity[self.__one_qubit_gates[idx]] = fidelity
        else:
            self.__one_qubit_gates_fidelity = gates_fidelity

    def register_one_qubit_rule(self, one_qubit_rule: Union[str, callable]):
        """ register one-qubit gate decompostion rule

        Args:
            one_qubit_rule(callable): decompostion rule, you can define your self rule function or use one of
                [zyz_rule, zxz_rule, hrz_rule, xyx_rule, ibmq_rule, u3_rule].
        """
        assert isinstance(one_qubit_rule, (str, FunctionType)), \
            TypeError("Unsupport Type, should be one of [string, Callable].")
        self.__one_qubit_rule = one_qubit_rule

    def register_two_qubit_rule_map(self, two_qubit_rule: Union[str, callable], source: GateType):
        """ register rule which transforms from source gate into target gate

        Args:
            two_qubit_rule(callable): the transform rule
            source(GateType): the type of source gate
        """
        assert isinstance(source, GateType)
        self.two_qubit_rule_map[source] = two_qubit_rule
