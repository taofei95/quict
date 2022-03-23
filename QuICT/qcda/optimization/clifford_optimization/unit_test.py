import pytest

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate
from QuICT.core.gate import GateType
from QuICT.qcda.optimization.clifford_optimization import CliffordOptimization

pauli_list = [GateType.id, GateType.x, GateType.y, GateType.z]
clifford_single = [GateType.h, GateType.s, GateType.sdg, GateType.x, GateType.y, GateType.z]
clifford = clifford_single + [GateType.cx]


def test_partition():
    for n in range(2, 6):
        for _ in range(100):
            circuit = Circuit(n)
            circuit.random_append(10 * n, clifford)
            gates = CompositeGate(gates=circuit.gates)
            compute, pauli = CliffordOptimization.partition(gates)
            compute.extend(pauli.gates())
            np.set_printoptions(precision=3, suppress=True)
            assert np.allclose(gates.matrix(), pauli.phase * compute.matrix())


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
