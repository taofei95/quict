#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/6 3:55 下午
# @Author  : Wu Yu sen
# @File    : controlled_single_rotation.py

from QuICT.models import *

TOLERANCE = 1e-12

def _apply_ucr_n(angles, ucontrol_qubits, target_qubit, gate_class, rightmost_cnot):
    """
       Decomposition for an uniformly controlled single qubit rotation gate.
       Follows decomposition in arXiv:quant-ph/04504100v1
       For a rotation operator Ra it uses 2**len(ucontrol_qubits) CX and also
       2**len(ucontrol_qubits) single qubit rotations.
       Args:
           angles: rotation angles with the length of 2**len(ucontrol_qubits)
           ucontrol_qubits: control qubits
           target_qubit: target qubit
           gate_class: Rx, Ry or Rz
    """

    if len(ucontrol_qubits) == 0:
        if gate_class == GateType.Rx:
            gate = Rx(angles[0])
        elif gate_class == GateType.Ry:
            gate = Ry(angles[0])
        else:
            gate = Rz(angles[0])
        gate | target_qubit
    else:
        if rightmost_cnot[len(ucontrol_qubits)]:
            angles1 = []
            angles2 = []
            for lower_bits in range(2**(len(ucontrol_qubits)-1)):
                leading_0 = angles[lower_bits]
                leading_1 = angles[lower_bits + 2**(len(ucontrol_qubits)-1)]
                angles1.append((leading_0 + leading_1)/2.)
                angles2.append((leading_0 - leading_1)/2.)
        else:
            angles1 = []
            angles2 = []
            for lower_bits in range(2**(len(ucontrol_qubits)-1)):
                leading_0 = angles[lower_bits]
                leading_1 = angles[lower_bits + 2**(len(ucontrol_qubits)-1)]
                angles1.append((leading_0 - leading_1)/2.)
                angles2.append((leading_0 + leading_1)/2.)
        _apply_ucr_n(angles=angles1,
                     ucontrol_qubits=ucontrol_qubits[:-1],
                     target_qubit=target_qubit,
                     gate_class=gate_class,
                     rightmost_cnot=rightmost_cnot)
        # Very custom usage of Compute/CustomUncompute in the following.
        if rightmost_cnot[len(ucontrol_qubits)]:
            CX | (ucontrol_qubits[-1], target_qubit)
        else:
            CX | (ucontrol_qubits[-1], target_qubit)
        _apply_ucr_n(angles=angles2,
                     ucontrol_qubits=ucontrol_qubits[:-1],
                     target_qubit=target_qubit,
                     gate_class=gate_class,
                     rightmost_cnot=rightmost_cnot)
        # Next iteration on this level do the other cnot placement
        rightmost_cnot[len(ucontrol_qubits)] = (
            not rightmost_cnot[len(ucontrol_qubits)])

def get_gates(angles, ucontrol_qubits, target_qubit, gate_class, rightmost_cnot):
    if len(ucontrol_qubits) == 0:
        if gate_class == GateType.Rx:
            gate = Rx(angles[0])
        elif gate_class == GateType.Ry:
            gate = Ry(angles[0])
        else:
            gate = Rz(angles[0])
        gate | target_qubit
    else:
        if rightmost_cnot[len(ucontrol_qubits)]:
            angles1 = []
            angles2 = []
            for lower_bits in range(2**(len(ucontrol_qubits)-1)):
                leading_0 = angles[lower_bits]
                leading_1 = angles[lower_bits + 2**(len(ucontrol_qubits)-1)]
                angles1.append((leading_0 + leading_1)/2.)
                angles2.append((leading_0 - leading_1)/2.)
        else:
            angles1 = []
            angles2 = []
            for lower_bits in range(2**(len(ucontrol_qubits)-1)):
                leading_0 = angles[lower_bits]
                leading_1 = angles[lower_bits + 2**(len(ucontrol_qubits)-1)]
                angles1.append((leading_0 - leading_1)/2.)
                angles2.append((leading_0 + leading_1)/2.)
        _apply_ucr_n(angles=angles1,
                     ucontrol_qubits=ucontrol_qubits[:-1],
                     target_qubit=target_qubit,
                     gate_class=gate_class,
                     rightmost_cnot=rightmost_cnot)
        # Very custom usage of Compute/CustomUncompute in the following.
        if rightmost_cnot[len(ucontrol_qubits)]:
            CX | (ucontrol_qubits[-1], target_qubit)
        else:
            CX | (ucontrol_qubits[-1], target_qubit)
        _apply_ucr_n(angles=angles2,
                     ucontrol_qubits=ucontrol_qubits[:-1],
                     target_qubit=target_qubit,
                     gate_class=gate_class,
                     rightmost_cnot=rightmost_cnot)
        # Next iteration on this level do the other cnot placement
        rightmost_cnot[len(ucontrol_qubits)] = (
            not rightmost_cnot[len(ucontrol_qubits)])

class MultifoldControlledRotationModel(gateModel):

    def __call__(self, angle, gateclass = GateType.Rx):
        """
        Args:
            angle(list): the list of angle
            gateclass(gateType): the type of gate(Rx, Ry or Rz)
        Returns:
            MultifoldControlledRotationModel: the gate filled by parameters
        """
        self.pargs = [angle, gateclass]
        return self

    def build_gate(self, other):
        angles = self.pargs[0]
        gateclass = self.pargs[1]

        circuit = Circuit(other)
        qureg = circuit.qubits
        num_qubit = len(qureg)

        ucontrol_qubits = qureg[0: num_qubit - 1]

        target_qubit = qureg[-1]
        angles1 = []
        angles2 = []
        angles_series = angles
        for lower_bits in range(int(2 ** (len(ucontrol_qubits) - 1))):
            leading_0 = angles_series[lower_bits]
            print(2 ** (len(ucontrol_qubits) - 1) + lower_bits)
            leading_1 = angles_series[lower_bits + 2 ** (len(ucontrol_qubits) - 1)]
            angles1.append((leading_0 + leading_1) / 2.)
            angles2.append((leading_0 - leading_1) / 2.)
        rightmost_cnot = {}
        for i in range(len(ucontrol_qubits) + 1):
            rightmost_cnot[i] = True
        get_gates(angles=angles1,
                     ucontrol_qubits=ucontrol_qubits[:-1],
                     target_qubit=target_qubit,
                     gate_class=gateclass,
                     rightmost_cnot=rightmost_cnot)
        # Very custom usage of Compute/CustomUncompute in the following.
        CX | (ucontrol_qubits[-1], target_qubit)

        get_gates(angles=angles2,
                     ucontrol_qubits=ucontrol_qubits[:-1],
                     target_qubit=target_qubit,
                     gate_class=gateclass,
                     rightmost_cnot=rightmost_cnot)

        CX | (ucontrol_qubits[-1], target_qubit)

        return circuit

MultifoldControlledRotation = MultifoldControlledRotationModel()
