#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/20 4:53 下午
# @Author  : Han Yu
# @File    : gate_transform.py

import numpy as np

from .._synthesis import Synthesis
from .instruction_set import InstructionSet
from .special_set import USTCSet
from .transform_rule import TransformRule

from QuICT.core import *

def _two_qubit_transform(source, instruction_set):
    """ transform source gate into target gate with function

    if function is None, find the default rule

    Args:
        source(BasicGate): the source gate
        instruction_set(InstructionSet): the target instruction set

    Returns:
        TransformRule: the gate list which contains only 2-qubit gates in target instruction set and one qubit gates
    """
    return instruction_set.select_transform_rule(source)

def GateTransformModel(circuit, instruction_set = USTCSet):
    """ equivalently transfrom circuit into goal instruction set

    The algorithm will try two possible path, and return a better result:
    1. make continuous local two-qubit gates into SU(4), then decomposition with goal instruction set
    2. transform the two-qubit gate into instruction set one by one, then one-qubit gate

    Args:
        circuit(Circuit): the circuit to be transformed
        instruction_set(InstructionSet): the goal instruction set

    Returns:
        CompositeGate: the equivalent compositeGate with goal instruction set
    """
    compositeGate = CompositeGate(circuit.gates, with_copy = False)

    # trans 2-qubits gate
    compositeGateStep1 = CompositeGate()
    for gate in compositeGate:
        if gate.targets + gate.controls > 2:
            raise Exception("gate_transform only support 2-qubit and 1-qubit gate now.")
        if gate.type() != instruction_set.two_qubit_gate and gate.targets + gate.controls == 2:
            rule = _two_qubit_transform(gate, instruction_set)
            compositeGateStep1.extend(rule.transform(gate))
        else:
            compositeGateStep1.append(gate)
    # trans one qubit gate
    compositeGateStep2 = CompositeGate()
    unitaries = [np.identity(2, dtype=np.complex128) for _ in range(circuit.circuit_width())]
    for gate in compositeGateStep1:
        if gate.targets + gate.controls == 2:
            compositeGateStep2.extend(instruction_set.SU2_rule.transform(Unitary(unitaries[gate.targ]) & gate.targ))
            compositeGateStep2.extend(instruction_set.SU2_rule.transform(Unitary(unitaries[gate.carg]) & gate.carg))
            unitaries[gate.targ] = np.identity(2, dtype=np.complex128)
            unitaries[gate.carg] = np.identity(2, dtype=np.complex128)
            compositeGateStep2.append(gate)
        else:
            unitaries[gate.targ] = np.dot(gate.matrix.reshape(2, 2), unitaries[gate.targ])
    for i in range(circuit.circuit_width()):
        compositeGateStep2.extend(instruction_set.SU2_rule.transform(Unitary(unitaries[i]) & i))
    return compositeGateStep2

GateTransform = Synthesis(GateTransformModel)
