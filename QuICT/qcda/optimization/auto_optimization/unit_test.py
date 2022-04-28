import numpy as np
import pytest

from QuICT.core.gate import *
import pickle
from QuICT.qcda.optimization.auto_optimization.template import *
from QuICT.qcda.optimization.auto_optimization.phase_poly import PhasePolynomial
from QuICT.qcda.optimization.auto_optimization.dag import DAG
from QuICT.qcda.optimization.auto_optimization.auto_optimization import AutoOptimization
from QuICT.algorithm import SyntheticalUnitary
from QuICT.tools.interface import OPENQASMInterface
from QuICT.qcda.optimization.commutative_optimization import CommutativeOptimization
import os


def test_build_graph():
    for i, each in enumerate(hadamard_templates):
        print('hadamard template', i)
        each.template.get_circuit().draw(method='command')
        each.replacement.get_circuit().draw(method='command')
        mat_1 = SyntheticalUnitary.run(each.template.get_circuit())
        mat_2 = SyntheticalUnitary.run(each.replacement.get_circuit()) * np.exp(1j * each.phase)
        assert np.allclose(mat_1, mat_2), f'hadamard_templates {i} not equal'

    for i, each in enumerate(single_qubit_gate_templates):
        circ = Circuit(each.template.size)
        Rz(1.234) | circ(each.anchor)
        mat_1 = SyntheticalUnitary.run(circ)
        mat_2 = SyntheticalUnitary.run(each.template.get_circuit())
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

    # print()
    # circ.draw(method='command')

    dag = DAG(circ)
    assert AutoOptimization.reduce_hadamard_gates(dag) == 10, 'hadamard gates reduction failed'

    circ_optim = dag.get_circuit()
    # circ_optim.draw(method='command')
    mat_0 = SyntheticalUnitary.run(circ)
    mat_1 = SyntheticalUnitary.run(circ_optim)
    assert np.allclose(mat_0, mat_1), 'unitary changed after hadamard gates reduction'


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

    # print()
    # circ.draw(method='command')
    dag = DAG(circ)
    assert AutoOptimization.cancel_single_qubit_gates(dag) == 3, 'single qubit gates cancellation failed'

    circ_optim = dag.get_circuit()
    mat_0 = SyntheticalUnitary.run(circ)
    mat_1 = SyntheticalUnitary.run(circ_optim)
    assert np.allclose(mat_0, mat_1), 'unitary changed after single qubit gates cancellation'


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

    # print()
    # circ.draw(method='command')
    dag = DAG(circ)
    circ_optim = dag.get_circuit()

    assert AutoOptimization.cancel_two_qubit_gates(dag) == 8, 'cnot cancellation failed'
    # dag.get_circuit().draw(method='command')
    mat_0 = SyntheticalUnitary.run(circ)
    mat_1 = SyntheticalUnitary.run(circ_optim)
    assert np.allclose(mat_0, mat_1), 'unitary changed after cnot cancellation'


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
    # circ.draw(method='command')
    gates = DAG(circ)
    for i, pack in enumerate(AutoOptimization.enumerate_cnot_rz_circuit(gates)):
        _, _, node_cnt = pack
        assert node_cnt == correct_not_count[i], f'node count = {node_cnt} != {correct_not_count[i]}'


def check_circuit_optimization(circ: Circuit, label):
    circ_optim = AutoOptimization.execute(circ)
    mat_1 = SyntheticalUnitary.run(circ)
    mat_2 = SyntheticalUnitary.run(circ_optim)
    if not np.allclose(mat_1, mat_2):
        pickle.dump(circ.gates, open(f'circ_{label}.dat', 'wb'))
        assert False, f'test {label}: mat_1 and mat_2 not equal'


def test_random_circuit():
    n_qubit = 6
    n_gate = 300
    n_iter = 5
    print(f'random ciruit test: {n_qubit} qubits, {n_gate} gates, {n_iter} iterations.')
    for _ in range(n_iter):
        print('iteration', _)
        circ = Circuit(n_qubit)
        # circ.random_append(n_gate, typeList=[GATE_ID['H'], GATE_ID['CX'], GATE_ID['X'], GATE_ID['Rz']])
        check_circuit_optimization(circ, _)


def test_parameterize_all():
    bmk_path = '/home/longcheng/repo/optimizer/QFT_and_Adders/QFT8_before.qasm'
    circ = OPENQASMInterface.load_file(bmk_path).circuit
    # circ = Circuit(2)
    # Rz(-np.pi / 4) | circ(1)

    dag = DAG(circ)
    # AutoOptimization.parameterize_all(dag)
    # AutoOptimization.merge_rotations(dag)
    # dag.get_circuit().draw(filename='parameter.jpg')
    AutoOptimization.deparameterize_all(dag)
    AutoOptimization.parameterize_all(dag)
    dag.get_circuit().draw(filename='deparameter.jpg')

    circ_optim = dag.get_circuit()

    mat_1 = SyntheticalUnitary.run(circ)
    mat_2 = SyntheticalUnitary.run(circ_optim)
    print(mat_1 / mat_2)
    assert np.allclose(mat_1, mat_2), "unitary changed after parameterize_all"


def test_deparameterize():
    circ = Circuit(1)
    Rz(- np.pi / 4) | circ(0)
    g1, p1 = CommutativeOptimization.deparameterize(circ.gates[0])
    print(circ.gates[0].matrix, g1.matrix * np.exp(1j * p1), p1 / np.pi)


def test_benchmark():
    bmk_path = '/home/longcheng/repo/optimizer/QFT_and_Adders/'
    for filename in os.listdir(bmk_path):
        if filename.startswith('QFT') and filename.endswith('before.qasm'):
            print(filename)
            path = os.path.join(bmk_path, filename)

            circ = OPENQASMInterface.load_file(path).circuit
            # circ.draw(filename=f'{filename}_before.jpg')

            circ_optim = AutoOptimization.execute(circ, verbose=True)
            # circ_optim.draw(filename=f'{filename}_after.jpg')

            # print(len(circ_optim.gates), '/', len(circ.gates))


if __name__ == '__main__':
    pytest.main(['./unit_test.py'])
