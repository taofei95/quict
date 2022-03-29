import pytest
import random

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate
from QuICT.core.gate import GateType
from QuICT.qcda.optimization.clifford_optimization import CliffordOptimization

pauli_list = [GateType.id, GateType.x, GateType.y, GateType.z]
clifford_single = [GateType.h, GateType.s, GateType.sdg, GateType.x, GateType.y, GateType.z]
clifford = clifford_single + [GateType.cx]
compute_stage = [GateType.cx, GateType.h, GateType.s]


def test_partition():
    for n in range(2, 6):
        for _ in range(100):
            circuit = Circuit(n)
            circuit.random_append(10 * n, clifford)
            gates = CompositeGate(gates=circuit.gates)
            compute, pauli = CliffordOptimization.partition(gates)
            for gate in compute:
                assert gate.type in compute_stage
            compute.extend(pauli.gates())
            np.set_printoptions(precision=3, suppress=True)
            assert np.allclose(gates.matrix(), pauli.phase * compute.matrix())


def test_peephole():
    for n in range(2, 6):
        for _ in range(100):
            circuit = Circuit(n)
            circuit.random_append(10 * n, clifford)
            gates = CompositeGate(gates=circuit.gates)
            compute, pauli = CliffordOptimization.partition(gates)
            control_set = random.sample(list(range(n)), 2)
            meow = CliffordOptimization.symbolic_peephole_optimization(compute, control_set)
            meow.extend(pauli.gates())
            np.set_printoptions(precision=3, suppress=True)
            assert np.allclose(gates.matrix(), pauli.phase * meow.matrix())


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
