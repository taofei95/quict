import time
from itertools import chain

import numpy as np
import pytest

from QuICT.core.gate import *
import pickle
from QuICT.qcda.optimization.auto_optimization.template import *
from QuICT.qcda.optimization.auto_optimization import DAG, AutoOptimization
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
        circ = Circuit(each.template.width())
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
    CX | circ([0, 1])
    Rz(2) | circ(0)
    CX | circ([1, 0])
    Rz(np.pi) | circ(0)
    CX | circ([1, 0])
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

    # print()
    # circ.draw(method='command')
    dag = DAG(circ)
    circ_optim = dag.get_circuit()

    assert AutoOptimization.cancel_two_qubit_gates(dag) == 8, 'cnot cancellation failed'
    # dag.get_circuit().draw(method='command')
    mat_0 = SyntheticalUnitary.run(circ)
    mat_1 = SyntheticalUnitary.run(circ_optim)
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
    AutoOptimization.merge_rotations_upd(dag)
    circ_optim = dag.get_circuit()
    circ_optim.draw(method='command')

    mat_0 = SyntheticalUnitary.run(circ)
    mat_1 = SyntheticalUnitary.run(circ_optim)
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
    # circ.draw(method='command')
    gates = DAG(circ)
    for i, pack in enumerate(AutoOptimization.enumerate_cnot_rz_circuit(gates)):
        _, _, node_cnt = pack
        assert node_cnt == correct_not_count[i], f'node count = {node_cnt} != {correct_not_count[i]}'


def check_circuit_optimization(circ: Circuit, label):
    try:
        circ_optim = AutoOptimization.execute(circ, verbose=True)
    except Exception as e:
        pickle.dump(circ.gates, open(f'circ_{label}.dat', 'wb'))
        raise e

    mat_1 = SyntheticalUnitary.run(circ)
    mat_2 = SyntheticalUnitary.run(circ_optim)
    if not np.allclose(mat_1, mat_2):
        pickle.dump(circ.gates, open(f'circ_{label}.dat', 'wb'))
        assert False, f'test {label}: mat_1 and mat_2 not equal'


def test_parameterize_all():
    n_qubit = 6
    n_gate = 200

    support_gates = [GateType.h, GateType.cx, GateType.x, GateType.rz,
                     GateType.t, GateType.tdg, GateType.s, GateType.sdg, GateType.z]
    circ = Circuit(n_qubit)
    circ.random_append(n_gate, typelist=support_gates)
    dag = DAG(circ)
    AutoOptimization.parameterize_all(dag)
    circ_optim = dag.get_circuit()

    mat_1 = SyntheticalUnitary.run(circ)
    mat_2 = SyntheticalUnitary.run(circ_optim)
    assert np.allclose(mat_1, mat_2), "unitary changed after parameterize_all"

    AutoOptimization.deparameterize_all(dag)
    circ_optim = dag.get_circuit()
    mat_2 = SyntheticalUnitary.run(circ_optim)
    assert np.allclose(mat_1, mat_2), "unitary changed after parameterize_all"


def test_random_circuit():
    n_qubit = 10
    n_gate = 1000
    n_iter = 10
    print(f'random ciruit test: {n_qubit} qubits, {n_gate} gates, {n_iter} iterations.')
    # support_gates = [GateType.h, GateType.cx]
    support_gates = [GateType.h, GateType.cx, GateType.x, GateType.rz,
                     GateType.t, GateType.tdg, GateType.s, GateType.sdg, GateType.z]
    for _ in range(n_iter):
        print('iteration', _)
        circ = Circuit(n_qubit)

        circ.random_append(n_gate, typelist=support_gates)
        check_circuit_optimization(circ, _)


def test_benchmark():
    bmk_path = '/home/longcheng/repo/optimizer/Arithmetic_and_Toffoli/mod5_4_before_no_ccz.qasm'
    circ = OPENQASMInterface.load_file(bmk_path).circuit
    optim_circ = AutoOptimization.execute(circ, verbose=True)
    circ.draw(filename='original.jpg')
    optim_circ.draw(filename='quict2.jpg')

    # bmk_path = '/home/longcheng/repo/optimizer/Arithmetic_and_Toffoli/tof_3_after_light.qasm'
    # circ = OPENQASMInterface.load_file(bmk_path).circuit
    # circ.draw(filename='quipper.jpg')

    mat_1 = SyntheticalUnitary.run(circ)
    mat_2 = SyntheticalUnitary.run(optim_circ)
    assert np.allclose(mat_1, mat_2), "unitary changed after parameterize_all"

    print('=====================================')
    dag = DAG(optim_circ)
    AutoOptimization.parameterize_all(dag)
    dag.get_circuit().draw(filename='quict3.jpg')
    print(AutoOptimization.merge_rotations_upd(dag))
    time.sleep(1)


def test_circ_510():
    gates = pickle.load(open('circ_debug_510.dat', 'rb'))
    circ = Circuit(6)
    for g in gates:
        g | circ(list(chain(g.cargs, g.targs)))
    circ.draw(filename='510.jpg')
    # check_circuit_optimization(circ, 1234)
    dag = DAG(circ)
    AutoOptimization.reduce_hadamard_gates(dag)
    circ_before = dag.get_circuit()
    circ_before.draw(filename='before.jpg')
    AutoOptimization.merge_rotations_upd(dag)
    circ_after = dag.get_circuit()
    circ_after.draw(filename='after.jpg')

    mat_1 = SyntheticalUnitary.run(circ_before)
    mat_2 = SyntheticalUnitary.run(circ_after)
    assert np.allclose(mat_1, mat_2)


def test_circ_4():
    gates = pickle.load(open('circ_4.dat', 'rb'))
    circ = Circuit(3)
    for g in gates:
        g | circ(list(chain(g.cargs, g.targs)))
    circ.draw(method='command')

    print('after')
    # AutoOptimization.execute(circ).draw(method='command')
    dag = DAG(circ)
    AutoOptimization.parameterize_all(dag)
    dag.get_circuit().draw(method='command')
    AutoOptimization.merge_rotations_upd(dag)
    dag.get_circuit().draw(method='command')

    check_circuit_optimization(circ, 1234)


def test_circ_8():
    gates = pickle.load(open('circ_8.dat', 'rb'))
    circ = Circuit(6)
    for g in gates:
        g | circ(list(chain(g.cargs, g.targs)))
    circ.draw(filename='circ_8.jpg')
    AutoOptimization.execute(circ, verbose=True)


def test_circ_2():
    gates = pickle.load(open('circ_debug_2.dat', 'rb'))
    circ = Circuit(6)
    for g in gates:
        g | circ(list(chain(g.cargs, g.targs)))
    circ.draw(filename='circ_8.jpg')
    AutoOptimization.execute(circ, verbose=True)


def test_circ_234():
    gates = pickle.load(open('circ_234.dat', 'rb'))
    circ = Circuit(6)
    for g in gates:
        g | circ(list(chain(g.cargs, g.targs)))
    # circ.draw(filename='circ_8.jpg')
    AutoOptimization.execute(circ, verbose=True)


def test_circ_dd_1():
    gates = pickle.load(open('circ_dd_1.dat', 'rb'))
    circ = Circuit(6)
    for g in gates:
        g | circ(list(chain(g.cargs, g.targs)))
    # circ.draw(filename='circ_8.jpg')
    AutoOptimization.execute(circ, verbose=True)


def test_circ_9():
    gates = pickle.load(open('circ_debug_9.dat', 'rb'))
    circ = Circuit(6)
    for g in gates:
        g | circ(list(chain(g.cargs, g.targs)))
    # circ.draw(filename='circ_8.jpg')
    check_circuit_optimization(circ, 12345)


def test_circ_308():
    gates = pickle.load(open('circ_debug_308.dat', 'rb'))
    circ = Circuit(6)
    for g in gates:
        g | circ(list(chain(g.cargs, g.targs)))
    # circ.draw(filename='circ_8.jpg')
    check_circuit_optimization(circ, 12345)


def test_arithmetic_benchmark():
    # bmk_path = '/home/longcheng/repo/optimizer/Arithmetic_and_Toffoli/'
    bmk_path = '/home/longcheng/repo/optimizer/Arithmetic_and_Toffoli/'
    cnt = 0
    for filename in os.listdir(bmk_path):
        if filename.endswith('before_no_ccz.qasm'):
            cnt += 1
            print(filename)
            path = os.path.join(bmk_path, filename)

            circ = OPENQASMInterface.load_file(path).circuit
            if circ.size() > 1000:
                print('skipped: circuit too large')
                continue

            # circ.draw(filename=f'{filename}_before.jpg')

            circ_optim = AutoOptimization.execute(circ, verbose=True)
            # circ_optim.draw(filename=f'{filename}_after.jpg')

            # print(len(circ_optim.gates), '/', len(circ.gates))


if __name__ == '__main__':
    pytest.main(['./auto_optimization_unit_test.py'])
