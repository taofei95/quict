#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/11 12:13
# @Author  : Han Yu
# @File    : _exec_operator.py

"""

This file is to define the execute operator
to be inherited by more than one gate.

"""

def exec_single(gate, circuit):
    qState = circuit.qubits[gate.targ].qState
    if circuit.fidelity is not None:
        qState.deal_single_gate(gate, True, circuit.fidelity)
    else:
        qState.deal_single_gate(gate)

def exec_controlSingle(gate, circuit):
    qState0 = circuit.qubits[gate.carg].qState
    qState1 = circuit.qubits[gate.targ].qState
    qState0.merge(qState1)
    qState0.deal_control_single_gate(gate)

def exec_toffoli(gate, circuit):
    qState0 = circuit.qubits[gate.cargs[0]].qState
    qState1 = circuit.qubits[gate.cargs[1]].qState
    qState2 = circuit.qubits[gate.targ].qState
    qState0.merge(qState1)
    qState0.merge(qState2)
    qState0.deal_ccx_gate(gate)

def exec_measure(gate, circuit):
    qState0 = circuit.qubits[gate.targ].qState
    qState0.deal_measure_gate(gate)

def exec_reset(gate, circuit):
    qState0 = circuit.qubits[gate.targ].qState
    qState0.deal_reset_gate(gate)

def exec_barrier(gate, circuit):
    pass

def exec_swap(gate, circuit):
    qState0 = circuit.qubits[gate.targs[0]].qState
    qState1 = circuit.qubits[gate.targs[1]].qState
    qState0.merge(qState1)
    qState0.deal_swap_gate(gate)

def exec_perm(gate, circuit):
    targs = gate.targs
    if not isinstance(targs, list):
        qState = circuit.qubits[targs].qState
    else:
        qState = circuit.qubits[targs[0]].qState
    for i in range(1, gate.targets):
        new_qState = circuit.qubits[gate.targs[i]].qState
        qState.merge(new_qState)
    qState.deal_perm_gate(gate)

def exec_custom(gate, circuit):
    targs = gate.targs
    if not isinstance(targs, list):
        qState = circuit.qubits[targs].qState
    else:
        qState = circuit.qubits[targs[0]].qState
    for i in range(1, gate.targets):
        new_qState = circuit.qubits[targs[i]].qState
        qState.merge(new_qState)
    qState.deal_custom_gate(gate)

def exec_shorInit(gate, circuit):
    targs = gate.targs
    if not isinstance(targs, list):
        qState = circuit.qubits[targs].qState
    else:
        qState = circuit.qubits[targs[0]].qState
    for i in range(1, gate.targets):
        new_qState = circuit.qubits[gate.targs[i]].qState
        qState.merge(new_qState)
    qState.deal_shorInitial_gate(gate)

def exec_controlMulPerm(gate, circuit):
    targs = gate.targs
    if not isinstance(targs, list):
        qState = circuit.qubits[targs].qState
    else:
        qState = circuit.qubits[targs[0]].qState
    for i in range(1, gate.targets):
        new_qState = circuit.qubits[gate.targs[i]].qState
        qState.merge(new_qState)
    new_qState = circuit.qubits[gate.cargs[0]].qState
    qState.merge(new_qState)
    qState.deal_controlMulPerm_gate(gate)
