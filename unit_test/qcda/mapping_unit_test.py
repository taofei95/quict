import os
import pytest
from typing import List
from QuICT.core.layout import *
from QuICT.core.gate import *
from QuICT.core.circuit import *
from QuICT.qcda.mapping import MCTSMapping as Mapping
from QuICT.qcda.mapping.utility import CouplingGraph
from QuICT.tools.interface import OPENQASMInterface


def test_mapping():
    file_path = os.path.realpath(__file__)
    dir, _ = os.path.split(file_path)
    layout = Layout.load_file(f"{dir}/ibmq_casablanca.layout")
    qc = OPENQASMInterface.load_file(f"{dir}/example_test.qasm").circuit
    transformed_circuit = Mapping.execute(circuit=qc, layout=layout, init_mapping_method="naive")
    coupling_graph = CouplingGraph(coupling_graph=layout)
    gates: List[BasicGate] = transformed_circuit.gates
    for g in gates:
        if g.is_single() is not True:
            if g.type == GateType.swap:
                assert(coupling_graph.is_adjacent(g.targs[0], g.targs[1]))
            else:
                assert(coupling_graph.is_adjacent(g.targ, g.carg))
    ori_array = qc.matrix()
    trans_array = transformed_circuit.matrix()
    np.allclose(ori_array, trans_array)


if __name__ == "__main__":
    pytest.main("./mapping_test.py")
