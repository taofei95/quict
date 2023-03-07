import numpy as np
from random import choice

from QuICT.core.circuit import Circuit
from QuICT.core.utils import CLIFFORD_GATE_SET, GateType
from QuICT.core.utils.circuit_info import CircuitMode
from QuICT.qcda.optimization.circuit_partition import CircuitPartitionOptimization


def test_default_light_optimization():
    n_iter = 1
    n_qubit = 4
    n_block = 4
    n_gate = 20
    mode_list = [
        [CircuitMode.Misc, None],
        [CircuitMode.Arithmetic, [GateType.x, GateType.cx, GateType.ccx]],
        [CircuitMode.Clifford, CLIFFORD_GATE_SET],
        [CircuitMode.CliffordRz, CLIFFORD_GATE_SET + [GateType.t, GateType.tdg, GateType.rz]]
    ]

    qcda_heavy = CircuitPartitionOptimization(level='light', verbose=True, keep_phase=True)

    for _ in range(n_iter):
        circ = Circuit(n_qubit)
        for __ in range(n_block):
            mode, typelist = choice(mode_list)
            circ.random_append(n_gate, typelist)

        circ_l = qcda_heavy.execute(circ)
        assert np.allclose(circ_l.matrix(), circ.matrix())


def test_default_heavy_optimization():
    n_iter = 1
    n_qubit = 4
    n_block = 4
    n_gate = 20
    mode_list = [
        [CircuitMode.Misc, None],
        [CircuitMode.Arithmetic, [GateType.x, GateType.cx, GateType.ccx]],
        [CircuitMode.Clifford, CLIFFORD_GATE_SET],
        [CircuitMode.CliffordRz, CLIFFORD_GATE_SET + [GateType.t, GateType.tdg, GateType.rz]]
    ]

    qcda_heavy = CircuitPartitionOptimization(level='heavy', verbose=True, keep_phase=True)

    for _ in range(n_iter):
        circ = Circuit(n_qubit)
        for __ in range(n_block):
            mode, typelist = choice(mode_list)
            circ.random_append(n_gate, typelist)

        circ_l = qcda_heavy.execute(circ)
        assert np.allclose(circ_l.matrix(), circ.matrix())
