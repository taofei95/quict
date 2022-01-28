#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/14 10:54
# @Author  : Han Yu, Li Kaiqi
# @File    : _gateBuilder.py
import numpy as np
import random

from QuICT.core.qubit import Qubit, Qureg
from QuICT.core.gate import *
from QuICT.core.utils import GateType


GATE_TYPE_TO_CLASS = {
    GateType.h: HGate,
    GateType.s: SGate,
    GateType.sdg: SDaggerGate,
    GateType.x: XGate,
    GateType.y: YGate,
    GateType.z: ZGate,
    GateType.sx: SXGate,
    GateType.sy: SYGate,
    GateType.sw: SWGate,
    GateType.id: IDGate,
    GateType.u1: U1Gate,
    GateType.u2: U2Gate,
    GateType.u3: U3Gate,
    GateType.rx: RxGate,
    GateType.ry: RyGate,
    GateType.rz: RzGate,
    GateType.t: TGate,
    GateType.tdg: TDaggerGate,
    GateType.phase: PhaseGate,
    GateType.cz: CZGate,
    GateType.cx: CXGate,
    GateType.cy: CYGate,
    GateType.ch: CHGate,
    GateType.crz: CRzGate,
    GateType.cu1: CU1Gate,
    GateType.cu3: CU3Gate,
    GateType.fsim: FSimGate,
    GateType.Rxx: RxxGate,
    GateType.Ryy: RyyGate,
    GateType.Rzz: RzzGate,
    GateType.swap: SwapGate,
    GateType.cswap: CSwapGate,
    GateType.ccx: CCXGate,
    GateType.CCRz: CCRzGate,
    GateType.measure: MeasureGate,
    GateType.reset: ResetGate,
    GateType.barrier: BarrierGate,
    GateType.unitary: UnitaryGate,
    GateType.perm: PermGate,
    GateType.control_perm_detail: ControlPermMulDetailGate,
    GateType.perm_shift: PermShiftGate,
    GateType.control_perm_shift: ControlPermShiftGate,
    GateType.perm_mul: PermMulGate,
    GateType.control_perm_mul: ControlPermMulGate,
    GateType.perm_fx: PermFxGate,
    GateType.shor_init: ShorInitialGate,
    GateType.qft: QFTGate,
    GateType.iqft: IQFTGate
}


def build_gate(
    gate_type: GateType,
    qubits: list,
    params: list = None
):
    gate = GATE_TYPE_TO_CLASS[gate_type]()
    if params is not None:
        params = params if isinstance(params, list) else [params]
        gate = gate(*params)

    args_number = gate.controls + gate.targets
    if isinstance(qubits, Qubit):
        qubits = Qureg(qubits)
    elif isinstance(qubits, int):
        qubits = [qubits]
    assert len(qubits) == args_number

    if isinstance(qubits, Qureg):
        gate.assigned_qubits = qubits
    else:
        gate.cargs = qubits[:gate.controls]
        gate.targs = qubits[gate.controls:]

    return gate


def build_random_gate(
    gate_type: GateType,
    qubits: int,
    random_params: bool = False
):
    gate = GATE_TYPE_TO_CLASS[gate_type]()
    args_number = gate.controls + gate.targets
    choiced_qubits = random.sample(range(qubits), args_number)
    gate.cargs = choiced_qubits[:gate.controls]
    gate.targs = choiced_qubits[gate.controls:]

    if random_params and gate.params:
        gate.pargs = [np.random.uniform(0, 2 * np.pi, gate.params) for _ in range(gate.params)]

    return gate
