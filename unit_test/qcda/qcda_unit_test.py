import numpy as np
from random import choice

from QuICT.core.circuit import Circuit
from QuICT.core.utils import CLIFFORD_GATE_SET, GateType
from QuICT.core.utils.circuit_info import CircuitMode
from QuICT.qcda import QCDA


def test_default_light_optimization():
    n_iter = 5
    n_qubit = 8
    n_block = 4
    n_gate = 50
    mode_list = [
        [CircuitMode.Misc, None],
        [CircuitMode.Arithmetic, [GateType.x, GateType.cx, GateType.ccx]],
        [CircuitMode.Clifford, CLIFFORD_GATE_SET],
        [CircuitMode.CliffordRz, CLIFFORD_GATE_SET + [GateType.t, GateType.tdg, GateType.rz]]
    ]

    qcda_light = QCDA()
    qcda_light.add_default_optimization(level='light', keep_phase=True)

    for _ in range(n_iter):
        circ = Circuit(n_qubit)
        for __ in range(n_block):
            mode, typelist = choice(mode_list)
            circ.random_append(n_gate, typelist)

        circ_l = qcda_light.compile(circ)
        assert np.allclose(circ_l.matrix(), circ.matrix())


def test_default_heavy_optimization():
    n_iter = 5
    n_qubit = 8
    n_block = 4
    n_gate = 50
    mode_list = [
        [CircuitMode.Misc, None],
        [CircuitMode.Arithmetic, [GateType.x, GateType.cx, GateType.ccx]],
        [CircuitMode.Clifford, CLIFFORD_GATE_SET],
        [CircuitMode.CliffordRz, CLIFFORD_GATE_SET + [GateType.t, GateType.tdg, GateType.rz]]
    ]

    qcda_heavy = QCDA()
    qcda_heavy.add_default_optimization(level='heavy', keep_phase=True)

    for _ in range(n_iter):
        circ = Circuit(n_qubit)
        for __ in range(n_block):
            mode, typelist = choice(mode_list)
            circ.random_append(n_gate, typelist)

        circ_l = qcda_heavy.compile(circ)
        assert np.allclose(circ_l.matrix(), circ.matrix())
