#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:53 下午
# @Author  : Han Yu
# @File    : gate_transform.py

from .._synthesis import Synthesis
from .instruction_set import InstructionSet
from .special_set import IBMQSet
from .transform_rule import TransformRule

def GateTransformModel(circuit, instruction_set = IBMQSet):
    """ equivalently transfrom circuit into goal instruction set

    The algorithm will try two possible path, and return a better result:
    1. make continuous local two-qubit gates into SU(4), then decomposition with goal instruction set
    2. transform the two-qubit gate into instruction set one by one, then one-qubit gate

    Args:
        circuit(Circuit): the circuit to be transformed
        instruction_set(InstructionSet): the goal instruction set

    Returns:
        circuit(Circuit): the equivalent circuit with goal instruction set
    """
    pass


GateTransform = Synthesis(GateTransformModel)
