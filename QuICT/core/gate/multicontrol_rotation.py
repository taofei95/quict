from typing import *
import numpy as np

from QuICT.core.gate import *


class MultiControlRotation(object):
    def __init__(self, target_gate: GateType, param: float):
        assert target_gate in [GateType.ry, GateType.rz]

        self.gate_type = target_gate
        self.param = param
        self.gate_dict = {
            GateType.ry: Ry,
            GateType.rz: Rz,
        }
        self.cgate_dict = {
            GateType.ry: CRy,
            GateType.rz: CRz,
        }

    def __call__(self, control: list, target: int):
        n_ctrl = len(control)
        if n_ctrl == 0:
            gates = CompositeGate()
            self.gate_dict[self.gate_type](self.param) & target | gates
            return gates
        if n_ctrl == 1:
            cgate = self.cgate_dict[self.gate_type](self.param)
            gates = CompositeGate()
            cgate & [control[0], target] | gates
            return gates
        if n_ctrl >= 2:
            theta = self.param / 2
            cgate1 = self.cgate_dict[self.gate_type](theta)
            cgate2 = self.cgate_dict[self.gate_type](-theta)
            mct = MultiControlToffoli()
            mcr = MultiControlRotation(self.gate_type, theta)

            gates = CompositeGate()
            cgate1 & [control[-1], target] | gates
            mct(n_ctrl - 1) | gates
            cgate2 & [control[-1], target] | gates
            mct(n_ctrl - 1) | gates
            mcr(control[:-1], target) | gates
            return gates
