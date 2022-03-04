import pytest, random

import numpy as np

from QuICT.core.gate import build_gate, CompositeGate, GateType
from QuICT.qcda.synthesis.clifford.pauli_operator import PauliOperator

pauli_list = [GateType.id, GateType.x, GateType.y, GateType.z]
clifford_single = [GateType.h, GateType.s, GateType.x, GateType.y, GateType.z]

def test_conjugate_action():
    # test clifford_single
    for pauli in pauli_list:
        for clifford in clifford_single:
            p = PauliOperator([pauli])
            clifford_gate = build_gate(clifford, 0)
            p.conjugate_act(clifford_gate)

            pauli_gate = build_gate(pauli, 0)
            gates = CompositeGate()
            if clifford != GateType.s:
                with gates:
                    clifford_gate & 0
                    pauli_gate & 0
                    clifford_gate & 0
            else:
                clifford_inverse_gate = build_gate(GateType.sdg, 0)
                with gates:
                    clifford_inverse_gate & 0
                    pauli_gate & 0
                    clifford_gate & 0
            assert np.allclose(p.gates.matrix() * p.phase, gates.matrix())
    
    # test cx
    for pauli_0 in pauli_list:
        for pauli_1 in pauli_list:
            p = PauliOperator([pauli_0, pauli_1])
            cx_gate = build_gate(GateType.cx, [0, 1])
            p.conjugate_act(cx_gate)

            pauli_gate_0 = build_gate(pauli_0, 0)
            pauli_gate_1 = build_gate(pauli_1, 1)
            gates = CompositeGate()
            with gates:
                cx_gate & [0, 1]
                pauli_gate_0 & 0
                pauli_gate_1 & 1
                cx_gate & [0, 1]
            assert np.allclose(p.gates.matrix() * p.phase, gates.matrix())

def test_commutative():
    for n in range(1, 5):
        for _ in range(100):
            p_op = []
            q_op = []
            for _ in range(n):
                p_op.append(random.choice(pauli_list))
                q_op.append(random.choice(pauli_list))        
            p = PauliOperator(p_op)
            q = PauliOperator(q_op)
            commute = p.commute(q)
            assert commute == np.allclose(np.dot(p.gates.matrix(), q.gates.matrix()),
                                        np.dot(q.gates.matrix(), p.gates.matrix()))

def test_standardizer_generator():
    standard_list = [
        [GateType.x, GateType.z],
        [GateType.x, GateType.x],
        [GateType.x, GateType.id],
        [GateType.id, GateType.z],
        [GateType.id, GateType.id]
    ]
    for x in pauli_list:
        for z in pauli_list:
            if [x, z] in standard_list:
                continue
            else:
                for c in clifford_single:
                    clifford = build_gate(c, 0)
                    px = PauliOperator([x])
                    pz = PauliOperator([z])
                    print(px.operator, pz.operator)
                    px.conjugate_act(clifford)
                    pz.conjugate_act(clifford)
                    # if [px.operator[0], pz.operator[0]] in standard_list:
                    print(c, px.operator, px.phase, pz.operator, pz.phase)
                    print()
                print()


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
