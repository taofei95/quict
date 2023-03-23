import pickle

from QuICT.qcda.optimization.clifford_rz_optimization import CliffordRzOptimization
from QuICT.qcda.optimization.clifford_rz_optimization.template import *


def test_build_graph():
    for i, each in enumerate(generate_hadamard_gate_templates()):
        mat_1 = each.template.get_circuit().matrix()
        mat_2 = each.replacement.get_circuit().matrix() * np.exp(1j * each.phase)
        assert np.allclose(mat_1, mat_2), f'hadamard_templates {i} not equal'

    for i, each in enumerate(generate_single_qubit_gate_templates()):
        circ = Circuit(each.template.width())
        Rz(1.234) | circ(each.anchor)
        mat_1 = circ.matrix()
        mat_2 = each.template.get_circuit().matrix()
        assert np.allclose(mat_1.dot(mat_2), mat_2.dot(mat_1)), \
            f'single_qubit_gate_templates {i} does not commute with Rz'


def check_circuit_optimization(circ: Circuit, label, level='light'):
    try:
        AO = CliffordRzOptimization(level=level, verbose=False, keep_phase=True)
        circ_optim = AO.execute(circ)
    except Exception as e:
        pickle.dump(circ.gates, open(f'circ_{label}.dat', 'wb'))
        raise e

    mat_1 = circ.matrix()
    mat_2 = circ_optim.matrix()
    if not np.allclose(mat_1, mat_2):
        pickle.dump(circ.gates, open(f'circ_{label}.dat', 'wb'))
        assert False, f'test {label}: mat_1 and mat_2 not equal'


def test_light_optimization():
    n_qubit = 4
    n_gate = 100
    n_iter = 1

    print(f'random ciruit test: {n_qubit} qubits, {n_gate} gates, {n_iter} iterations.')
    support_gates = [GateType.h, GateType.cx, GateType.rz,
                     GateType.t, GateType.tdg, GateType.s, GateType.sdg, GateType.z, GateType.x,
                     GateType.ccx, GateType.ccz,
                     ]
    # support_gates = [GateType.h, GateType.cx, GateType.rz,]
    for _ in range(n_iter):
        print('iteration', _)
        circ = Circuit(n_qubit)

        circ.random_append(n_gate, typelist=support_gates, random_params=True)
        check_circuit_optimization(circ, _, level='light')


def test_heavy_optimization():
    n_qubit = 4
    n_gate = 100
    n_iter = 1

    print(f'random ciruit test: {n_qubit} qubits, {n_gate} gates, {n_iter} iterations.')
    support_gates = [GateType.h, GateType.cx, GateType.rz,
                     GateType.t, GateType.tdg, GateType.s, GateType.sdg, GateType.z, GateType.x,
                     GateType.ccx, GateType.ccz,
                     ]
    # support_gates = [GateType.h, GateType.cx, GateType.rz,]
    for _ in range(n_iter):
        print('iteration', _)
        circ = Circuit(n_qubit)

        circ.random_append(n_gate, typelist=support_gates, random_params=True)
        check_circuit_optimization(circ, _, level='heavy')


def test_disabling_optimize_toffoli():
    n_qubit = 4
    n_gate = 50
    support_gates = [GateType.h, GateType.cx, GateType.rz,
                     GateType.t, GateType.tdg, GateType.s, GateType.sdg, GateType.z, GateType.x,
                     GateType.ccx, GateType.ccz,
                     ]

    for level in ['light', 'heavy']:
        circ = Circuit(n_qubit)
        circ.random_append(n_gate, typelist=support_gates)
        cnt = sum([g.type == GateType.ccx or g.type == GateType.ccz for g in circ.gates])
        circ_optim = CliffordRzOptimization(level=level, optimize_toffoli=False, keep_phase=True).execute(circ)
        cnt_optim = sum([g.type == GateType.ccx or g.type == GateType.ccz for g in circ_optim.gates])

        assert cnt == cnt_optim, f'ccx/ccz changed in {level} mode'

        mat_1 = circ.matrix()
        mat_2 = circ_optim.matrix()
        assert np.allclose(mat_1, mat_2), 'matrix changed'
