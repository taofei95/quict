import pytest
from QuICT.core import *
import pickle
from .template import *
from .phase_poly import PhasePolynomial
from .dag import DAG
from .auto_optimization import AutoOptimization
from QuICT.algorithm import SyntheticalUnitary


def test_build_graph():
    for i, each in enumerate(hadamard_templates):
        print('hadamard template', i)
        each.template.get_circuit().draw(method='command')
        each.replacement.get_circuit().draw(method='command')

    for i, each in enumerate(single_qubit_gate_templates):
        print('single qubit template', i)
        each.template.get_circuit().draw(method='command')

    for i, each in enumerate(cnot_ctrl_template):
        print('cnot targ template', i)
        each.template.get_circuit().draw(method='command')

    for i, each in enumerate(cnot_targ_template):
        print('cnot targ template', i)
        each.template.get_circuit().draw(method='command')


def test_enumerate_sub_circuit():
    gate_list_1 = [[H, 0], [Rz(1), 1], [CX, (0, 1)], [H, 0], [Rz(2), 1], [CX, (1, 0)], [Rz(0), 0]]
    circ = get_circuit_from_list(2, gate_list_1)
    dag = DAG(circ)

    for prev_node, succ_node, node_cnt in dag.enumerate_sub_circuit({'rz', 'cx'}):
        print(node_cnt)


def test_phase_poly():
    # TODO random benchmarking
    # gate_list_1 = [[Rz, 1], [CX, (0, 1)], [Rz, 1], [CX, (0, 1)], [Rz, 1]]
    # circ = get_circuit_from_list(2, gate_list_1)
    circ = Circuit(2)
    Rz(1) | circ(1)
    CX | circ((0, 1))
    Rz(2) | circ(1)
    CX | circ((0, 1))
    Rz(3) | circ(1)

    circ.draw(method='command')
    poly = PhasePolynomial(DAG(circ))
    poly.get_circuit().draw(method='command')

    circ = Circuit(2)

    Rz(1) | circ(0)
    CX | circ((0, 1))
    Rz(2) | circ(1)
    X | circ(0)
    CX | circ((0, 1))
    Rz(1) | circ(0)

    circ.draw(method='command')
    PhasePolynomial(DAG(circ)).get_circuit().draw(method='command')


def test_compare_circuit():
    pattern = Circuit(2)
    Rz(0) | pattern(0)
    CX | pattern((0, 1))
    H | pattern(1)

    replace = Circuit(2)
    CX | replace((1, 0))

    circ = Circuit(2)
    CX | circ((0, 1))
    Rz(0) | circ(0)
    CX | circ((0, 1))
    H | circ(1)
    Rz(0) | circ(1)
    CX | circ((1, 0))
    H | circ(0)

    print()
    circ.draw(method='command')

    pattern = DAG(pattern)
    circ = DAG(circ)
    replace = DAG(replace)
    temp = OptimizationTemplate(pattern, replace, anchor=0)

    temp.replace_all(circ)
    circ.get_circuit().draw(method='command')

    # for each in circ.topological_sort():
    #     print(each.gate.qasm_name)
    #     print(pattern.compare_circuit(each, 0, flag_enabled=False))


def test_reduce_hadamard_gate():
    circ = Circuit(3)

    H | circ(0)
    S | circ(0)
    H | circ(0)
    H | circ(1)
    S_dagger | circ(1)
    H | circ(1)
    CX | circ((0, 1))
    H | circ(0)
    H | circ(1)
    CX | circ((0, 1))
    H | circ(0)
    H | circ(1)
    CX | circ((1, 0))
    H | circ(0)
    H | circ(1)
    S | circ(1)
    CX | circ((2, 1))
    S_dagger | circ(1)
    H | circ(1)
    S_dagger | circ(0)
    CX | circ((2, 0))
    S | circ(0)
    H | circ(0)

    print()
    circ.draw(method='command')

    dag = DAG(circ)
    print(AutoOptimization.reduce_hadamard_gates(dag))

    print()
    dag.get_circuit().draw(method='command')


def test_cancel_single_qubit_gate():
    circ = Circuit(2)
    Rz(1) | circ(0)
    Rz(-1) | circ(1)
    CX | circ((0, 1))
    Rz(2) | circ(0)
    CX | circ((1, 0))
    Rz(np.pi) | circ(0)
    CX | circ((1, 0))
    Rz(-3) | circ(0)

    print()
    circ.draw(method='command')
    dag = DAG(circ)

    print(AutoOptimization.cancel_single_qubit_gates(dag))
    dag.get_circuit().draw(method='command')


def test_cancel_two_qubit_gate():
    circ = Circuit(3)
    CX | circ((0, 1))
    CX | circ((0, 2))
    CX | circ((0, 1))
    H | circ(2)
    CX | circ((2, 1))
    H | circ(2)
    CX | circ((0, 2))
    CX | circ((0, 1))
    H | circ(1)
    CX | circ((0, 2))
    CX | circ((1, 2))
    CX | circ((0, 2))
    H | circ(1)
    CX | circ((0, 1))

    print()
    circ.draw(method='command')
    dag = DAG(circ)

    print(AutoOptimization.cancel_two_qubit_gates(dag))
    dag.get_circuit().draw(method='command')

# def test_build_graph_2():
#     print()
#     n_qubit, tpl, rpl = [2, [[H, 0], [H, 1], [CX, (0, 1)], [H, 0], [H, 1]], [[CX, (1, 0)]]]
#     # tpl_circ = get_circuit_from_list(n_qubit, tpl)
#     rpl_circ = get_circuit_from_list(n_qubit, rpl)
#     DAG(rpl_circ).get_circuit().draw(method='command')


def test_enumerate_cnot_rz_circuit():
    circ = Circuit(2)
    Rz(1) | circ(0)
    CX | circ((0, 1))
    Rz(2) | circ(0)
    CX | circ((1, 0))
    H | circ(1)
    Rz(np.pi) | circ(1)
    CX | circ((0, 1))
    Rz(-1) | circ(1)
    CX | circ((0, 1))
    Rz(np.pi) | circ(1)

    circ.draw(method='command')
    gates = DAG(circ)
    for _, _, node_cnt in AutoOptimization.enumerate_cnot_rz_circuit(gates):
        print(node_cnt)


def test_merge_rotations():
    circ = Circuit(2)
    Rz(1) | circ(0)
    CX | circ((0, 1))
    Rz(2) | circ(0)
    CX | circ((1, 0))
    H | circ(1)
    Rz(np.pi) | circ(1)
    CX | circ((0, 1))
    Rz(-1) | circ(1)
    CX | circ((0, 1))
    Rz(np.pi) | circ(1)

    print()
    circ.draw(method='command')

    # print('direct')
    # PhasePolynomial(DAG(circ)).get_circuit().draw(method='command')

    dag = DAG(circ)
    print('merge: ', AutoOptimization.merge_rotations(dag))
    dag.get_circuit().draw(method='command')


def test_enumerate_cnot_rz_circuit_2():
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

    print()
    circ.draw(method='command')

    # print('direct')
    # PhasePolynomial(DAG(circ)).get_circuit().draw(method='command')

    dag = DAG(circ)

    print('what the fuck')
    # DAG.copy_sub_circuit(list(zip(dag.start_nodes, [0] * 3)), list(zip(dag.end_nodes, [0] * 3))).get_circuit().draw(method='command')

    print('merge: ', AutoOptimization.merge_rotations(dag))
    dag.get_circuit().draw(method='command')


def test_auto_optimization():
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

    circ.draw(method='command')
    # dag = DAG(circ)
    circ_a = AutoOptimization.execute(circ)
    print('after auto optim')
    circ_a.draw(method='command')


def check_circuit_optimization(circ: Circuit, label):
    # circ.draw(filename=f'before_{label}.jpg')
    pickle.dump(circ.gates, open(f'circ_{label}.dat', 'wb'))

    circ_optim = AutoOptimization.execute(circ)
    # circ_optim.draw(filename=f'after_{label}.jpg')

    # DONE check equiv
    mat_1 = SyntheticalUnitary.run(circ)
    mat_2 = SyntheticalUnitary.run(circ_optim)
    if not np.allclose(mat_1, mat_2):
        assert False, f'test {label}: mat_1 and mat_2 not equal'


def test_random_circuit():
    n_qubit = 8
    n_gate = 1000
    n_iter = 1
    for _ in range(n_iter):
        print('iter', _)
        circ = Circuit(n_qubit)
        circ.random_append(n_gate, typeList=[GATE_ID['H'], GATE_ID['CX'], GATE_ID['X'], GATE_ID['Rz']])
        check_circuit_optimization(circ, _)


def test_circ_6():
    gates = pickle.load(open('circ_debug_6.dat', 'rb'))
    circ = Circuit(3)
    for g in gates:
        g | circ(g.affectArgs)
    circ.draw(method='command')

    AutoOptimization.execute(circ)


def test_circ_0():
    gates = pickle.load(open('circ_debug_0.dat', 'rb'))
    circ = Circuit(3)
    for g in gates:
        if g.qasm_name == 'h' and g.targ == 2:
            continue
        g | circ(g.affectArgs)
    circ.draw(method='command')

    check_circuit_optimization(circ, 123)

    # AutoOptimization.execute(circ).draw(filename='single.jpg')


    # AutoOptimization.reduce_hadamard_gates(dag)
    # AutoOptimization.merge_rotations(dag)
    # dag.get_circuit().draw(filename='single.jpg')


def test_circ_1():
    gates = pickle.load(open('circ_debug_1.dat', 'rb'))
    circ = Circuit(3)
    for g in gates:
        if g.qasm_name == 'h' and g.targ == 2:
            continue
        g | circ(g.affectArgs)

    check_circuit_optimization(circ, 123)


def test_circ_1_1():
    gates = pickle.load(open('circ_debug_1_1.dat', 'rb'))
    circ = Circuit(3)
    for g in gates:
        if g.qasm_name == 'h' and g.targ == 2:
            continue
        g | circ(g.affectArgs)

    check_circuit_optimization(circ, 123)


def test_circ_30():
    gates = pickle.load(open('circ_debug_30.dat', 'rb'))
    circ = Circuit(6)
    for g in gates:
        if g.qasm_name == 'h' and g.targ == 2:
            continue
        g | circ(g.affectArgs)

    check_circuit_optimization(circ, 123)


def test_circ_57():
    gates = pickle.load(open('circ_debug_57.dat', 'rb'))
    circ = Circuit(6)
    for g in gates:
        g | circ(g.affectArgs)
    circ.draw(method='command')

    check_circuit_optimization(circ, 123)


def test_circ_36_1():
    gates = pickle.load(open('circ_debug_57.dat', 'rb'))
    circ = Circuit(6)
    for g in gates:
        g | circ(g.affectArgs)
    circ.draw(method='command')

    check_circuit_optimization(circ, 123)


if __name__ == '__main__':
    pytest.main(['./unit_test.py'])
