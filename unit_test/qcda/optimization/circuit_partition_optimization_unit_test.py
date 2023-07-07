import numpy as np
from random import choice

from QuICT.core.circuit import Circuit
from QuICT.core.utils import CLIFFORD_GATE_SET, GateType, CircuitMode
from QuICT.qcda.optimization.circuit_partition import CircuitPartitionOptimization


def test_default_light_optimization():
    n_iter = 3
    n_qubit = 6
    n_block = 6
    n_gate = 100
    mode_list = [
        [CircuitMode.Misc, None],
        [CircuitMode.Arithmetic, [GateType.x, GateType.cx, GateType.ccx]],
        [CircuitMode.Clifford, CLIFFORD_GATE_SET],
        [CircuitMode.CliffordRz, CLIFFORD_GATE_SET + [GateType.t, GateType.tdg, GateType.rz]]
    ]

    cp_light = CircuitPartitionOptimization(level='light', verbose=True, keep_phase=True)

    for _ in range(n_iter):
        circ = Circuit(n_qubit)
        for __ in range(n_block):
            mode, typelist = choice(mode_list)
            print(__, mode)
            circ.random_append(n_gate, typelist)

        circ_l = cp_light.execute(circ)
        assert np.allclose(circ_l.matrix(), circ.matrix())


def test_default_heavy_optimization():
    n_iter = 3
    n_qubit = 4
    n_block = 4
    n_gate = 40
    mode_list = [
        [CircuitMode.Misc, None],
        [CircuitMode.Arithmetic, [GateType.x, GateType.cx, GateType.ccx]],
        [CircuitMode.Clifford, CLIFFORD_GATE_SET],
        [CircuitMode.CliffordRz, CLIFFORD_GATE_SET + [GateType.t, GateType.tdg, GateType.rz]]
    ]

    cp_heavy = CircuitPartitionOptimization(level='heavy', verbose=True, keep_phase=True)

    for _ in range(n_iter):
        circ = Circuit(n_qubit)
        for __ in range(n_block):
            mode, typelist = choice(mode_list)
            print(__, mode)
            circ.random_append(n_gate, typelist)

        circ_h = cp_heavy.execute(circ)
        assert np.allclose(circ_h.matrix(), circ.matrix())


if __name__ == '__main__':
    test_default_light_optimization()
