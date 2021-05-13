from QuICT.core import *
from QuICT.qcda.optimization.template_optimization.template_optimization import TemplateOptimization
from QuICT.qcda.optimization.template_optimization.template_matching import TemplateMatching, TemplateSubstitution, MaximalMatches
from QuICT.qcda.optimization.template_optimization.template_matching.dagdependency import DAGDependency, circuit_to_dagdependency
import networkx as nx
import numpy as np

if __name__ == '__main__':
    # template in the paper
    circuit_T = Circuit(5)

    CX | circuit_T([3, 0])
    X | circuit_T(4)
    Z | circuit_T(0)
    CX | circuit_T([4, 2])
    CX | circuit_T([0, 1])
    CX | circuit_T([3, 4])
    CX | circuit_T([1, 2])
    X | circuit_T(1)
    CX | circuit_T([1, 0])
    X | circuit_T(1)
    CX | circuit_T([1, 2])
    CX | circuit_T([0, 3])

    dag_T = DAGDependency()
    dag_T = circuit_to_dagdependency(circuit_T)

    #dag_T._graph.draw(filename='T.jpg', layout=nx.shell_layout)
    for node_id in dag_T.get_nodes():
        node = dag_T.get_node(node_id)
        print(node.node_id, node.gate.type(), node.cargs, node.targs)

    # circuit in the paper
    circuit_C = Circuit(8)

    CX | circuit_C([6, 7])
    CX | circuit_C([7, 5])
    CX | circuit_C([6, 7])
    CCX | circuit_C([7, 6, 5])
    CX | circuit_C([6, 7])
    CX | circuit_C([1, 4])
    CX | circuit_C([6, 3])
    CX | circuit_C([3, 4])
    CX | circuit_C([4, 5])
    CX | circuit_C([0, 5])
    Z | circuit_C(3)
    X | circuit_C(4)
    CX | circuit_C([4, 3])
    CX | circuit_C([3, 1])
    X | circuit_C(4)
    CX | circuit_C([1, 2])
    CX | circuit_C([3, 1])
    CX | circuit_C([3, 5])
    CX | circuit_C([3, 6])
    X | circuit_C(3)
    CX | circuit_C([4, 5])

    dag_C = DAGDependency()
    dag_C = circuit_to_dagdependency(circuit_C)

    #dag_C._graph.draw(filename='C.jpg', layout=nx.shell_layout)
    for node_id in dag_C.get_nodes():
        node = dag_C.get_node(node_id)
        print(node.node_id, node.gate.type(), node.cargs, node.targs)

    template_m = TemplateMatching(dag_C, dag_T)
    all_list_first_match_q = \
        template_m._list_first_match_new(template_m.circuit_dag_dep.get_node(6),
                                    template_m.template_dag_dep.get_node(0),
                                    5)

    heuristics_qubits = template_m._explore_circuit(node_id_c=6,
                                                node_id_t=0,
                                                n_qubits_t=5,
                                                length=2)

    print(all_list_first_match_q, heuristics_qubits)
    template_m.run_template_matching()
    matches = template_m.match_list

    maximal = MaximalMatches(matches)
    maximal.run_maximal_matches()
    max_matches = maximal.max_match_list

    for match in max_matches:
        print(match.match, match.qubit, '\n')

    # Not a correct use since circuit_T is not identity, only to
    # check if the code could run in case of a non-trivial input
    circuit_opt = TemplateOptimization.execute(circuit_C, [circuit_T])
    circuit_opt.print_information()
