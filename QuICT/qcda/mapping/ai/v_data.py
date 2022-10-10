#!/usr/bin/env python3

"""This file generates some random circuits used for validation of trained RL agent.
"""

import os
import os.path as osp

from QuICT.core import *
from QuICT.core.layout.layout import Layout
from QuICT.core.utils import GateType

# from QuICT.qcda.mapping.mcts import MCTSMapping

data_dir = osp.dirname(osp.realpath(__file__))
data_dir = osp.join(data_dir, "data")
data_dir = osp.join(data_dir, "v_data")

if not osp.exists(data_dir):
    os.makedirs(data_dir)

circ_num_each_topo = 5

for topo_name in ["grid_4x4", "grid_3x3", "ibmq_lima"]:
    print(f"Starting processing {topo_name}...")
    topo_dir = osp.dirname(osp.abspath(__file__))
    topo_dir = osp.join(topo_dir, "data")
    topo_dir = osp.join(topo_dir, "topo")
    topo_path = osp.join(topo_dir, f"{topo_name}.json")
    topo = Layout.load_file(topo_path)
    q = topo.qubit_number
    # mcts_mapper = MCTSMapping(layout=topo)
    for i in range(circ_num_each_topo):
        print(".", end="")
        circ = Circuit(q)
        g = 10 * (i + 1)
        circ.random_append(rand_size=g, typelist=[GateType.cx])
        qasm = circ.qasm()
        qasm_path = osp.join(data_dir, f"{topo_name}_{i}.qasm")
        with open(qasm_path, "w") as f:
            f.write(qasm)

        # mapped_circ = mcts_mapper.execute(circuit=circ)
        # mapped_qasm = mapped_circ.qasm()
        # qasm_path = osp.join(data_dir, f"mapped_{topo_name}_{i}.qasm")
        # with open(qasm_path, "w") as f:
        #     f.write(mapped_qasm)

    print()
