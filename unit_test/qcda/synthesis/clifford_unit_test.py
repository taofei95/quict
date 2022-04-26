import pytest
import random

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate, GateType
from QuICT.qcda.utility import PauliOperator
from QuICT.qcda.synthesis.clifford.clifford_synthesizer import CliffordUnidirectionalSynthesizer,\
    CliffordBidirectionalSynthesizer

clifford_single = [GateType.h, GateType.s, GateType.sdg, GateType.x, GateType.y, GateType.z]
clifford = clifford_single + [GateType.cx]


def test_uni_disentangle_one_qubit():
    for n in range(2, 6):
        for _ in range(10):
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
        for _ in range(10):
            circuit = Circuit(n)
            circuit.random_append(10 * n, clifford)
            gates = CompositeGate(gates=circuit.gates)
            CUS = CliffordUnidirectionalSynthesizer(strategy='greedy')
            # CUS = CliffordUnidirectionalSynthesizer(strategy='random')
            circ_syn = CUS.execute(circuit)
            gates_remain = gates.inverse()
            gates_remain.extend(circ_syn.gates)
            # np.set_printoptions(precision=3, suppress=True)
            assert np.allclose(gates_remain.matrix(), gates_remain.matrix()[0][0] * np.eye(2 ** n))


def test_bi_disentangle_one_qubit():
    for n in range(2, 6):
        for _ in range(10):
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
        for _ in range(10):
            circuit = Circuit(n)
            circuit.random_append(10 * n, clifford)
            gates = CompositeGate(gates=circuit.gates)
            CBS = CliffordBidirectionalSynthesizer(
                qubit_strategy='greedy',
                pauli_strategy='random',
                shots=10,
                multiprocess=False,
                process=12,
                chunksize=64
            )
            circ_syn = CBS.execute(circuit)
            gates_remain = gates.inverse()
            gates_remain.extend(circ_syn.gates)
            # np.set_printoptions(precision=3, suppress=True)
            assert np.allclose(gates_remain.matrix(), gates_remain.matrix()[0][0] * np.eye(2 ** n))


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
