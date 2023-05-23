from random import sample

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.tools.circuit_library import CircuitLib
from QuICT.qcda.optimization.template_optimization.template_matching import (
    ForwardMatch, MatchingDAGCircuit)
from QuICT.qcda.optimization.template_optimization.template_optimization import \
    TemplateOptimization


def get_circ():
    circ = Circuit(8)
    CX | circ([6, 7])
    CX | circ([7, 5])
    CX | circ([6, 7])
    CCX | circ([7, 6, 5])
    CX | circ([6, 7])
    CX | circ([1, 4])
    CX | circ([6, 3])
    CX | circ([3, 4])
    CX | circ([4, 5])
    CX | circ([0, 5])
    Z | circ(3)
    X | circ(4)
    CX | circ([4, 3])
    X | circ(4)
    CX | circ([3, 1])
    CX | circ([1, 2])
    CX | circ([3, 1])
    CX | circ([3, 5])
    CX | circ([3, 6])
    X | circ(3)
    CX | circ([4, 5])

    template = Circuit(5)
    CX | template([3, 0])
    X | template(4)
    Z | template(0)
    CX | template([4, 2])
    CX | template([0, 1])
    CX | template([3, 4])
    CX | template([1, 2])
    X | template(1)
    CX | template([1, 0])
    X | template(1)
    CX | template([1, 2])
    CX | template([0, 3])

    return circ, template


def test_forward_matching():
    circ, template = get_circ()
    circ_dag = MatchingDAGCircuit(circ)
    template_dag = MatchingDAGCircuit(template)

    c_node_id = 6
    t_node_id = 0
    qubit_mapping = [3, 4, 5, 6, 7]

    res = ForwardMatch.execute(circ_dag, template_dag, c_node_id, t_node_id, qubit_mapping)
    ans = [(0, 6), (4, 7), (6, 8), (2, 10), (7, 11), (8, 12), (9, 13), (11, 18), (10, 20)]
    assert res == ans


def test_random_circuit():
    gates = [GateType.x, GateType.cx, GateType.ccx, GateType.h, GateType.s, GateType.t,
             GateType.sdg, GateType.tdg]

    n_qubits = 4
    template_list = CircuitLib().get_template_circuit()
    n_templates = 10

    for n_gates in range(20, 101, 20):
        circ = Circuit(n_qubits)
        circ.random_append(n_gates, typelist=gates)
        TO = TemplateOptimization(template_typelist=sample(template_list, n_templates))
        circ_optim = TO.execute(circ)

        mat_1 = circ.matrix()
        mat_2 = circ_optim.matrix()
        assert np.allclose(mat_1, mat_2)
