import os
from typing import List

from QuICT.core.layout import *
from QuICT.core.gate import *
from QuICT.core.circuit import *
from QuICT.qcda.mapping import MCTSMapping
from QuICT.qcda.mapping.utility import CouplingGraph
from QuICT.tools.interface import OPENQASMInterface


def test_mapping():
    file_path = os.path.realpath(__file__)
    dir, _ = os.path.split(file_path)
    layout = Layout.load_file(f"{dir}/example/ibmq_casablanca.layout")
    qc = OPENQASMInterface.load_file(f"{dir}/example/example_test.qasm").circuit

    MCTS = MCTSMapping(layout=layout, init_mapping_method="naive")
    transformed_circuit = MCTS.execute(circuit=qc)
    coupling_graph = CouplingGraph(coupling_graph=layout)
    gates: List[BasicGate] = transformed_circuit.gates
    for g in gates:
        if g.is_single() is not True:
            if g.type == GateType.swap:
                assert(coupling_graph.is_adjacent(g.targs[0], g.targs[1]))
            else:
                assert(coupling_graph.is_adjacent(g.targ, g.carg))
#     qasm = OPENQASMInterface.load_circuit(transformed_circuit)
#     qasm.output_qasm(f"{dir}/output_circuit.qasm")
#     print("The original circuit size is {}. After mapping, its size is {}."
#           .format(qc.size(), transformed_circuit.size()))
#     CouplingGraph(coupling_graph=layout).draw(file_path=f"{dir}/coupling_graph.jpg")
#     qc.draw(method="matp", filename=f"{dir}/original_circuit.jpg")
#     Check if the number of single qubit gates and two qubit gates(except SWAP gates) remains the same
#     transformed_circuit.draw(method="matp",
#                              filename=f"{dir}/transformed_circuit.jpg")
#     print([qc.count_1qubit_gate(), transformed_circuit.count_1qubit_gate()] )
#     print([qc.count_2qubit_gate(), transformed_circuit.count_2qubit_gate()] )
