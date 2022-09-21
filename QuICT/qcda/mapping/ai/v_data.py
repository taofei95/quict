"""This file generates some random circuits used for validation of trained RL agent.
"""

import os
import os.path as osp

from QuICT.core import *
from QuICT.core.layout.layout import Layout
from QuICT.core.utils import GateType
from QuICT.qcda.mapping.ai.data_factory import DataFactory
from QuICT.qcda.mapping.mcts import MCTSMapping

factory = DataFactory(max_qubit_num=30, max_gate_num=50)

data_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.join(data_dir, "data")
data_dir = osp.join(data_dir, "v_data")

if not osp.exists(data_dir):
    os.makedirs(data_dir)

circ_num_each_topo = 2


for topo_name in factory.topo_names:
    print(f"Starting processing {topo_name}...")
    topo_path = osp.join(factory._topo_dir, f"{topo_name}.json")
    topo = Layout.load_file(topo_path)
    q = topo.qubit_number
    mcts_mapper = MCTSMapping(layout=topo)
    for i in range(circ_num_each_topo):
        print(".", end="")
        circ = Circuit(q)
        g = 20 * (i + 1)
        circ.random_append(rand_size=g, typelist=[GateType.crz])
        qasm = circ.qasm()
        qasm_path = osp.join(data_dir, f"{topo_name}_{i}.qasm")
        with open(qasm_path, "w") as f:
            f.write(qasm)

        mapped_circ = mcts_mapper.execute(circuit=circ)
        mapped_qasm = mapped_circ.qasm()
        qasm_path = osp.join(data_dir, f"mapped_{topo_name}_{i}.qasm")
        with open(qasm_path, "w") as f:
            f.write(mapped_qasm)

    print()
