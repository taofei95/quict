#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:53 下午
# @Author  : Han Yu
# @File    : gate_transform.py

from .._synthesis import Synthesis
from .instruction_set import InstructionSet
from .special_set import IBMQSet
from .transform_rule import TransformRule

class GateTransformModel(Synthesis):
    """ synthesis gates into goal instruction set



    """

    def __init__(self):
        super().__init__()
        self.instruction_set = IBMQSet
        self.circuit = None

    def __call__(self, circuit, instruction_set = None):
        """
        Args:
            instruction_set(InstructionSet): the goal InstructionSet
        Returns:
            GateTransform: model filled by the instruction_set.
        """
        self.circuit = circuit
        if instruction_set is not None:
            self.instruction_set = instruction_set
        return self

    def build_gate(self):
        """ overloaded the function "build_gate"

        """
        return self.transform_circuit(self.circuit, self.instruction_set)

    @staticmethod
    def transform_circuit(circuit, instructionSet):
        """ equivalently transfrom circuit into goal instruction set

        The algorithm will try two possible path, and return a better result:
        1. make continuous local two-qubit gates into SU(4), then decomposition with goal instruction set
        2. transform the two-qubit gate into instruction set one by one, then one-qubit gate

        Args:
            circuit(Circuit): the circuit to be transformed
            instructionSet(InstructionSet): the goal instruction set

        Returns:
            circuit(Circuit): the equivalent circuit with goal instruction set
        """
        pass


GateTransform = GateTransformModel()
