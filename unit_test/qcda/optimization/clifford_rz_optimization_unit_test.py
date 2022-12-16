import pickle
import time

from QuICT.qcda.optimization.clifford_rz_optimization import (
    DAG, CliffordRzOptimization)
from QuICT.qcda.optimization.clifford_rz_optimization.template import *


def test_build_graph():
    for i, each in enumerate(generate_hadamard_gate_templates()):
        print('hadamard template', i)
        each.replacement.get_circuit().draw(method='command')
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


def test_reduce_hadamard_gates():
    circ = Circuit(3)

    H | circ(0)
    S | circ(0)
    H | circ(0)
    H | circ(1)
    S_dagger | circ(1)
    H | circ(1)
    CX | circ([0, 1])
    H | circ(0)
    H | circ(1)
    CX | circ([0, 1])
    H | circ(0)
    H | circ(1)
    CX | circ([1, 0])
    H | circ(0)
    H | circ(1)
    S | circ(1)
    CX | circ([2, 1])
    S_dagger | circ(1)
    H | circ(1)
    S_dagger | circ(0)
    CX | circ([2, 0])
    S | circ(0)
    H | circ(0)

    dag = DAG(circ)
    assert CliffordRzOptimization().reduce_hadamard_gates(dag) == 10, 'hadamard gates reduction failed'

    circ_optim = dag.get_circuit()
    # circ_optim.draw(method='command')
    mat_0 = circ.matrix()
    mat_1 = circ_optim.matrix()
    assert np.allclose(mat_0, mat_1), 'unitary changed after hadamard gates reduction'


def test_cancel_single_qubit_gate():
    circ = Circuit(2)
    Rz(1) | circ(0)
    Rz(-1) | circ(1)
    CX | circ([0, 1])
    Rz(2) | circ(0)
    CX | circ([1, 0])
    Rz(np.pi) | circ(0)
    CX | circ([1, 0])
    Rz(-3) | circ(0)

    dag = DAG(circ)
    assert CliffordRzOptimization().cancel_single_qubit_gates(dag) == 3, 'single qubit gates cancellation failed'

    circ_optim = dag.get_circuit()

    mat_0 = circ.matrix()
    mat_1 = circ_optim.matrix()
    assert np.allclose(mat_0, mat_1), 'unitary changed after single qubit gates cancellation'


def test_cancel_two_qubit_gate():
    circ = Circuit(3)
    CX | circ([0, 1])
    CX | circ([0, 2])
    CX | circ([0, 1])
    H | circ(2)
    CX | circ([2, 1])
    H | circ(2)
    CX | circ([0, 2])
    CX | circ([0, 1])
    H | circ(1)
    CX | circ([0, 2])
    CX | circ([1, 2])
    CX | circ([0, 2])
    H | circ(1)
    CX | circ([0, 1])

    dag = DAG(circ)
    # print(AutoOptimization.cancel_two_qubit_gates(dag))
    assert CliffordRzOptimization().cancel_two_qubit_gates(dag) == 8, 'cnot cancellation failed'
    circ_optim = dag.get_circuit()

    # circ.draw(filename='a.jpg')
    # circ_optim.draw(filename='b.jpg')

    mat_0 = circ.matrix()
    mat_1 = circ_optim.matrix()
    assert np.allclose(mat_0, mat_1), 'unitary changed after cnot cancellation'


def test_merge_rotations():
    circ = Circuit(2)
    Rz(1) | circ(0)
    CX | circ([0, 1])
    Rz(2) | circ(0)
    CX | circ([1, 0])
    H | circ(1)
    Rz(np.pi) | circ(1)
    CX | circ([0, 1])
    Rz(-1) | circ(1)
    CX | circ([0, 1])
    Rz(np.pi) | circ(1)

    print()
    circ.draw(method='command')
    dag = DAG(circ)
    CliffordRzOptimization().merge_rotations(dag)
    circ_optim = dag.get_circuit()
    circ_optim.draw(method='command')

    mat_0 = circ.matrix()
    mat_1 = circ_optim.matrix()
    assert np.allclose(mat_0, mat_1), 'unitary changed after merge rotations'

    time.sleep(1)


def test_enumerate_cnot_rz_circuit():
    circ = Circuit(3)
    H | circ(0)
    H | circ(1)
    H | circ(2)
    Rz(1) | circ(1)
    Rz(2) | circ(2)
    CX | circ([1, 0])
    Rz(-1) | circ(0)
    CX | circ([1, 2])
    CX | circ([0, 1])
    H | circ(2)
    CX | circ([1, 2])
    CX | circ([0, 1])
    Rz(3) | circ(1)
    H | circ(0)
    H | circ(1)

    correct_not_count = [6, 3]
    gates = DAG(circ)
    for i, pack in enumerate(CliffordRzOptimization()._enumerate_cnot_rz_circuit(gates)):
        _, _, node_cnt = pack
        assert node_cnt == correct_not_count[i], f'node count = {node_cnt} != {correct_not_count[i]}'


def check_circuit_optimization(circ: Circuit, label, level='light'):
    try:
        AO = CliffordRzOptimization(level=level, verbose=False)
        circ_optim = AO.execute(circ)
    except Exception as e:
        pickle.dump(circ.gates, open(f'circ_{label}.dat', 'wb'))
        raise e

    mat_1 = circ.matrix()
    mat_2 = circ_optim.matrix()
    if not np.allclose(mat_1, mat_2):
        pickle.dump(circ.gates, open(f'circ_{label}.dat', 'wb'))
        assert False, f'test {label}: mat_1 and mat_2 not equal'


def test_parameterize_all():
    n_qubit = 6
    n_gate = 200

    support_gates = [GateType.h, GateType.cx, GateType.x, GateType.rz,
                     GateType.t, GateType.tdg, GateType.s, GateType.sdg, GateType.z]
    circ = Circuit(n_qubit)
    circ.random_append(n_gate, typelist=support_gates, random_params=True)
    dag = DAG(circ)
    CliffordRzOptimization().parameterize_all(dag)
    circ_optim = dag.get_circuit()

    mat_1 = circ.matrix()
    mat_2 = circ_optim.matrix()
    assert np.allclose(mat_1, mat_2), "unitary changed after parameterize_all"

    CliffordRzOptimization().deparameterize_all(dag)
    circ_optim = dag.get_circuit()
    mat_2 = circ_optim.matrix()
    assert np.allclose(mat_1, mat_2), "unitary changed after parameterize_all"


def test_ccx():
    circ = Circuit(6)
    circ.random_append(10, typelist=[GateType.ccx])

    AO = CliffordRzOptimization(optimize_toffoli=True)
    circ_optim = AO.execute(circ)
    circ_optim.draw(filename='ccx_after.jpg')

    mat_0 = circ.matrix()
    mat_1 = circ_optim.matrix()

    assert np.allclose(mat_0, mat_1), 'unitary changed'


def test_float_rz():
    circ = Circuit(2)

    CX | circ([0, 1])
    Rz(1) | circ(1)
    CX | circ([0, 1])
    Rz(2) | circ(0)
    Rz(3) | circ(1)
    CX | circ([1, 0])

    dag = DAG(circ)
    assert CliffordRzOptimization().float_rotations(dag) == 2, 'float_rotations not correct'

    circ_optim = dag.get_circuit()
    mat_1 = circ.matrix()
    mat_2 = circ_optim.matrix()
    assert np.allclose(mat_1, mat_2), "unitary changed after float rz"


def test_gate_preserving_template():
    for each in generate_gate_preserving_rewrite_template():
        print('template,', each.param_order)
        circ = each.template.get_circuit()
        circ_optim = each.replacement.get_circuit()

        # circ.draw(method='command')
        # circ_optim.draw(method='command')

        mat_1 = circ.matrix()
        mat_2 = circ_optim.matrix()
        assert np.allclose(mat_1, mat_2), "gate preserving template different"


def test_gate_reducing_template():
    for each in generate_gate_reducing_rewrite_template():
        print('template,', each.param_order)
        circ = each.template.get_circuit()
        circ_optim = each.replacement.get_circuit()

        # circ.draw(method='command')
        # circ_optim.draw(method='command')

        mat_1 = circ.matrix()
        mat_2 = circ_optim.matrix()
        assert np.allclose(mat_1, mat_2), "gate preserving template different"


def test_regrettable_replace():
    circ = Circuit(2)
    X | circ(0)
    CX | circ([1, 0])
    H | circ(0)
    H | circ(1)
    CX | circ([0, 1])
    H | circ(0)
    H | circ(1)
    Rz | circ(1)
    dag = DAG(circ)

    hadamard_templates = generate_hadamard_gate_templates()

    for node in dag.topological_sort():
        mapping = hadamard_templates[3].compare((node, -1))
        if mapping:
            orig, undo = hadamard_templates[3].regrettable_replace(mapping)
            circ1 = dag.get_circuit()
            hadamard_templates[3].undo_replace(orig, undo)
            circ2 = dag.get_circuit()

            mat_1 = circ1.matrix()
            mat_2 = circ2.matrix()
            assert np.allclose(mat_1, mat_2), "regrettable_replace or undo_replace not correct"

            return


def test_disabling_optimize_toffoli():
    n_qubit = 6
    n_gate = 200
    support_gates = [GateType.h, GateType.cx, GateType.rz,
                     GateType.t, GateType.tdg, GateType.s, GateType.sdg, GateType.z, GateType.x,
                     GateType.ccx, GateType.ccz,
                     ]

    for level in ['light', 'heavy']:
        circ = Circuit(n_qubit)
        circ.random_append(n_gate, typelist=support_gates)
        cnt = sum([g.type == GateType.ccx or g.type == GateType.ccz for g in circ.gates])
        circ_optim = CliffordRzOptimization(level=level, optimize_toffoli=False).execute(circ)
        cnt_optim = sum([g.type == GateType.ccx or g.type == GateType.ccz for g in circ_optim.gates])

        assert cnt == cnt_optim, f'ccx/ccz changed in {level} mode'

        mat_1 = circ.matrix()
        mat_2 = circ_optim.matrix()
        assert np.allclose(mat_1, mat_2), 'matrix changed'


def test_random_circuit():
    n_qubit = 6
    n_gate = 200
    n_iter = 5

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
        check_circuit_optimization(circ, _, level='heavy')
