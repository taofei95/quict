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

    # def __call__(self, control: list, target: int):
    #     n_ctrl = len(control)
    #     if n_ctrl == 0:
    #         gates = CompositeGate()
    #         self.gate_dict[self.gate_type](self.param) | gates(target)
    #         return gates
    #     if n_ctrl == 1:
    #         cgate = self.cgate_dict[self.gate_type](self.param)
    #         gates = CompositeGate()
    #         cgate | gates([control[0], target])
    #         return gates
    #     if n_ctrl >= 2:
    #         theta = self.param / 2
    #         cgate1 = self.cgate_dict[self.gate_type](theta)
    #         cgate2 = self.cgate_dict[self.gate_type](-theta)
    #         mct = MultiControlToffoli()
    #         mcr = MultiControlRotation(self.gate_type, theta)

    #         gates = CompositeGate()
    #         cgate1 | gates([control[-1], target])
    #         mct(n_ctrl - 1) | gates(control)
    #         cgate2 | gates([control[-1], target])
    #         mct(n_ctrl - 1) | gates(control)
    #         mcr(control[:-1], target) | gates
    #         return gates

    def __call__(self, control: int):
        gates = CompositeGate()
        if control == 0:
            self.gate_dict[self.gate_type](self.param) | gates(0)
            return gates
        if control == 1:
            cgate = self.cgate_dict[self.gate_type](self.param)
            cgate | gates([0, 1])
            return gates

        theta = self.param / 2
        cgate1 = self.cgate_dict[self.gate_type](theta)
        cgate2 = self.cgate_dict[self.gate_type](-theta)
        change_bit = self._get_change_bit(control)
        q_state = [0] * control
        related_qubits = []
        for i in range(len(change_bit)):
            print("related_qubits", related_qubits)
            q_state[change_bit[i]] = 1 - q_state[change_bit[i]]
            control_bits = self._get_control_bits(q_state)

            if len(related_qubits) > 0:
                diff_set = set(related_qubits) - set(control_bits)
                print(set(related_qubits), set(control_bits))
                if diff_set:
                    CX | gates([list(diff_set)[0], control_bits[-1]])
                    related_qubits.remove(list(diff_set)[0])
                    related_qubits.remove(control_bits[-1])
                else:
                    diff_set = set(control_bits) - set(related_qubits)
                    CX | gates([list(diff_set)[0], control_bits[-1]])
                    related_qubits.append(list(diff_set)[0])
                print("CX", [list(diff_set)[0], control_bits[-1]])
            else:
                if len(control_bits) == 2:
                    CX | gates(control_bits)
                    print("CX", control_bits)
                    for bit in control_bits:
                        related_qubits.append(bit)

            cgate = cgate1 if i % 2 == 0 else cgate2
            cgate | gates([control_bits[-1], control])
            print("CRy", [control_bits[-1], control])
            # cgate | gates([control - 1 - list(reversed(q_state)).index(1), control])
        return gates

    def _get_change_bit(self, control):
        if control == 0:
            return []
        if control == 1:
            return [0]
        pre_change_bit = self._get_change_bit(control - 1)
        return pre_change_bit + [control - 1] + pre_change_bit

    def _get_control_bits(self, q_state):
        control_bits = []
        for i in range(len(q_state)):
            if q_state[i] == 1:
                control_bits.append(i)
        return control_bits
