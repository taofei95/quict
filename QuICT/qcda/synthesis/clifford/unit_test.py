import pytest
import random

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import build_gate, CompositeGate, GateType
from QuICT.qcda.synthesis.clifford.pauli_operator import PauliOperator
from QuICT.qcda.synthesis.clifford.clifford_synthesizer import CliffordUnidirectionalSynthesizer,\
    CliffordBidirectionalSynthesizer

pauli_list = [GateType.id, GateType.x, GateType.y, GateType.z]
clifford_single = [GateType.h, GateType.s, GateType.sdg, GateType.x, GateType.y, GateType.z]
clifford = clifford_single + [GateType.cx]


def test_conjugate_action():
    # test clifford_single
    for pauli in pauli_list:
        for clifford in clifford_single:
            p = PauliOperator([pauli])
            clifford_gate = build_gate(clifford, 0)
            p.conjugate_act(clifford_gate)

            pauli_gate = build_gate(pauli, 0)
            gates = CompositeGate()
            if clifford == GateType.s:
                clifford_inverse_gate = build_gate(GateType.sdg, 0)
                with gates:
                    clifford_inverse_gate & 0
                    pauli_gate & 0
                    clifford_gate & 0
            elif clifford == GateType.sdg:
                clifford_inverse_gate = build_gate(GateType.s, 0)
                with gates:
                    clifford_inverse_gate & 0
                    pauli_gate & 0
                    clifford_gate & 0
            else:
                with gates:
                    clifford_gate & 0
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
            p = PauliOperator.random(n)
            q = PauliOperator.random(n)
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
            for c1 in [GateType.id] + clifford_single:
                for c2 in [GateType.id] + clifford_single:
                    found = False
                    clifford_1 = build_gate(c1, 0)
                    clifford_2 = build_gate(c2, 0)
                    px = PauliOperator([x])
                    pz = PauliOperator([z])
                    if c1 == GateType.id and c2 == GateType.id:
                        print(px.operator, pz.operator)
                    px.conjugate_act(clifford_1)
                    pz.conjugate_act(clifford_1)
                    px.conjugate_act(clifford_2)
                    pz.conjugate_act(clifford_2)
                    if [px.operator[0], pz.operator[0]] in standard_list:
                        print(c1, c2, px.operator, px.phase, pz.operator, pz.phase)
                        found = True
                        break
                if found is True:
                    break
            print()


def test_swap():
    for pauli_0 in pauli_list:
        for pauli_1 in pauli_list:
            p = PauliOperator([pauli_0, pauli_1])
            cx_gate = build_gate(GateType.cx, [0, 1])
            p.conjugate_act(cx_gate)
            p.conjugate_act(cx_gate & [1, 0])
            p.conjugate_act(cx_gate)
            assert p.operator[0] == pauli_1 and p.operator[1] == pauli_0


def test_disentangler_fixed():
    x_op = [GateType.x]
    z_op = [GateType.z]
    for x in pauli_list:
        for z in pauli_list:
            x_op.append(x)
            z_op.append(z)
    pauli_x = PauliOperator(x_op)
    pauli_z = PauliOperator(z_op)
    target = random.randint(0, 16)
    disentangler = PauliOperator.disentangler(pauli_x, pauli_z, target)
    for gate in disentangler:
        pauli_x.conjugate_act(gate)
        pauli_z.conjugate_act(gate)
    assert pauli_x.phase == 1 and pauli_z.phase == 1
    for i in range(pauli_x.width):
        if i == target:
            assert pauli_x.operator[i] == GateType.x and pauli_z.operator[i] == GateType.z
        else:
            assert pauli_x.operator[i] == GateType.id and pauli_z.operator[i] == GateType.id


def test_disentangler_random():
    for n in range(1, 50):
        for _ in range(100):
            pauli_x, pauli_z = PauliOperator.random_anti_commutative_pair(n)
            target = random.randint(0, n - 1)
            disentangler = PauliOperator.disentangler(pauli_x, pauli_z, target)
            for gate in disentangler:
                pauli_x.conjugate_act(gate)
                pauli_z.conjugate_act(gate)
            assert pauli_x.phase == 1 and pauli_z.phase == 1
            for i in range(pauli_x.width):
                if i == target:
                    assert pauli_x.operator[i] == GateType.x and pauli_z.operator[i] == GateType.z
                else:
                    assert pauli_x.operator[i] == GateType.id and pauli_z.operator[i] == GateType.id


def test_uni_disentangle_one_qubit():
    for n in range(2, 10):
        for _ in range(100):
            circuit = Circuit(n)
            circuit.random_append(10 * n, clifford)
            gates = CompositeGate(gates=circuit.gates)
            target = random.randint(0, n - 1)
            disentangler = CliffordUnidirectionalSynthesizer.disentangle_one_qubit(gates, n, target)
            gates_next = gates.inverse()
            gates_next.extend(disentangler)

            x_op = [GateType.id for _ in range(n)]
            z_op = [GateType.id for _ in range(n)]
            x_op[target] = GateType.x
            z_op[target] = GateType.z
            pauli_x = PauliOperator(x_op)
            pauli_z = PauliOperator(z_op)
            for gate in gates_next:
                pauli_x.conjugate_act(gate)
                pauli_z.conjugate_act(gate)
            assert pauli_x.phase == 1 and pauli_z.phase == 1
            assert pauli_x.operator == x_op and pauli_z.operator == z_op


def test_unidirectional():
    for n in range(2, 6):
        for _ in range(100):
            circuit = Circuit(n)
            circuit.random_append(10 * n, clifford)
            gates = CompositeGate(gates=circuit.gates)
            gates_syn = CliffordUnidirectionalSynthesizer.execute(circuit, strategy='greedy')
            # gates_syn = CliffordUnidirectionalSynthesizer.execute(circuit, strategy='random')
            gates_remain = gates.inverse()
            gates_remain.extend(gates_syn)
            # np.set_printoptions(precision=3, suppress=True)
            assert np.allclose(gates_remain.matrix(), gates_remain.matrix()[0][0] * np.eye(2 ** n))


def test_bi_disentangle_one_qubit():
    for n in range(2, 10):
        for _ in range(100):
            circuit = Circuit(n)
            circuit.random_append(10 * n, clifford)
            gates = CompositeGate(gates=circuit.gates)
            target = random.randint(0, n - 1)
            p1, p2 = PauliOperator.random_anti_commutative_pair(n)
            left, right = CliffordBidirectionalSynthesizer.disentangle_one_qubit(gates, target, p1, p2)
            gates_next = right.inverse()
            gates_next.extend(gates.inverse())
            gates_next.extend(left)

            x_op = [GateType.id for _ in range(n)]
            z_op = [GateType.id for _ in range(n)]
            x_op[target] = GateType.x
            z_op[target] = GateType.z
            pauli_x = PauliOperator(x_op)
            pauli_z = PauliOperator(z_op)
            for gate in gates_next:
                pauli_x.conjugate_act(gate)
                pauli_z.conjugate_act(gate)
            assert pauli_x.phase == 1 and pauli_z.phase == 1
            assert pauli_x.operator == x_op and pauli_z.operator == z_op


def test_bidirectional():
    for n in range(2, 6):
        for _ in range(100):
            circuit = Circuit(n)
            circuit.random_append(10 * n, clifford)
            gates = CompositeGate(gates=circuit.gates)
            gates_syn = CliffordBidirectionalSynthesizer.execute(circuit, qubit_strategy='random')
            gates_remain = gates.inverse()
            gates_remain.extend(gates_syn)
            # np.set_printoptions(precision=3, suppress=True)
            assert np.allclose(gates_remain.matrix(), gates_remain.matrix()[0][0] * np.eye(2 ** n))


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
