from typing import *
import numpy as np

from QuICT.core.gate import *
from QuICT.core import Circuit
from QuICT.tools.exception.core import TypeError, ValueError


class MultiControlRotation(object):
    def __init__(self, target_gate: BasicGate):
        assert target_gate.type in [GateType.rx, GateType.ry, GateType.rz], TypeError(
            "MultiControlRotation.target_gate",
            [GateType.rx, GateType.ry, GateType.rz],
            target_gate.type,
        )
        self.target_gate = target_gate
        self.param = target_gate.pargs[0]

    def __call__(self, control: list, target: int):
        assert target not in control
        n_ctrl = len(control)
        if n_ctrl == 0:
            gates = CompositeGate()
            self.target_gate & target | gates
            return gates
        if n_ctrl == 1:
            cgate_dict = {
                GateType.rx: CRx,
                GateType.ry: CRy,
                GateType.rz: CRz,
            }
            cgate = cgate_dict[self.target_gate.type](self.param)
            gates = CompositeGate()
            cgate & [control[0], target] | gates
            return gates


if __name__ == "__main__":
    
    #mcr = MultiControlRotation(Ry(0.5))
    #gates = mcr([0], 1)
    #print(gates)
    cir = Circuit(2)
    # gates | cir
    H | cir(0)
    
    cir.draw(filename="1.jpg")
