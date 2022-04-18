import pytest

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate, GateType
from QuICT.qcda.optimization.symbolic_clifford_optimization import SymbolicCliffordOptimization

pauli_list = [GateType.id, GateType.x, GateType.y, GateType.z]
clifford_single = [GateType.h, GateType.s, GateType.sdg, GateType.x, GateType.y, GateType.z]
clifford = clifford_single + [GateType.cx]
compute_stage = [GateType.cx, GateType.h, GateType.s]


def test_partition():
    for n in range(2, 6):
        for _ in range(10):
            circuit = Circuit(n)
            circuit.random_append(20 * n, clifford)
            gates = CompositeGate(gates=circuit.gates)
            compute, pauli = SymbolicCliffordOptimization.partition(gates)
            for gate in compute:
                assert gate.type in compute_stage
            compute.extend(pauli.gates())
            # np.set_printoptions(precision=3, suppress=True)
            assert np.allclose(gates.matrix(), pauli.phase * compute.matrix())


def test():
    for n in range(2, 6):
        for _ in range(10):
            circuit = Circuit(n)
            circuit.random_append(20 * n, clifford)
            gates = CompositeGate(gates=circuit.gates)
            gates_opt = SymbolicCliffordOptimization.execute(gates)
            # np.set_printoptions(precision=3, suppress=True)
            assert np.allclose(gates.matrix(), gates_opt.matrix())


if __name__ == '__main__':
    pytest.main(["./unit_test.py"])
