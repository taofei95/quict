#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/8/22 2:39
# @Author  : Han Yu
# @File    : synthesis.py

from QuICT.qcda.synthesis import GateDecomposition, GateTransform

class Synthesis(object):
    """ Synthesize gates with some basic gates

    In general, these basic gates are one-qubit gates and two-qubit gates.

    A Optimization means several qubit optimization methods, which would be executed sequentially.
    If the methods are not assigned, a default sequence will be given.
    """
    def __init__(self, instruction=None, methods=None):
        """
        Args:
            instruction(InstructionSet): target InstructionSet
            methods(list, optional): a list of used methods
        """
        assert instruction is not None, ValueError('No InstructionSet provided for Synthesis')

        if methods is not None:
            self.methods = methods
        else:
            self.methods = []
            decomposition = GateDecomposition()
            transform = GateTransform(instruction)
            self.methods.append(decomposition)
            self.methods.append(transform)

    def execute(self, circuit):
        """
        Synthesize the circuit to the InstructionSet with the given methods

        Args:
            circuit(CompositeGate/Circuit): the target CompositeGate or Circuit
        """
        for method in self.methods:
            circuit = method.execute(circuit)

        return circuit
