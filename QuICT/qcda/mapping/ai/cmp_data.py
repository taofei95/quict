#!/usr/bin/env python3

import os
import os.path as osp
import re
from QuICT.core import *
from QuICT.tools import Logger
from QuICT.qcda.mapping import MCTSMapping
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

logger = Logger(tag="CmpDataGen")


def map_circ(circ: Circuit, layout: Layout) -> Circuit:
    mapper = MCTSMapping(layout=layout, bp_num=100, sim_cnt=5)
    result = mapper.execute(circ)
    return result


def main():
    data_dir = osp.abspath(osp.dirname(__file__))
    data_dir = osp.join(data_dir, "data")
    topo_dir = osp.join(data_dir, "topo")
    layouts = {}
    for _, _, file_names in os.walk(topo_dir):
        for file_name in file_names:
            layout = Layout.load_file(osp.join(topo_dir, file_name))
            layouts[layout.name] = layout

    qasm_paths = []
    v_data_dir = osp.join(data_dir, "v_data")
    for _, _, file_names in os.walk(v_data_dir):
        for file_name in file_names:
            if file_name.startswith("mapped"):
                continue
            qasm_path = osp.join(v_data_dir, file_name)
            qasm_paths.append(qasm_path)
    for qasm_path in qasm_paths:
        logger.info(qasm_path)
        circ = OPENQASMInterface.load_file(qasm_path).circuit
        pattern = re.compile(
            r"(/[_0-9a-zA-Z]+)*/([0-9a-zA-Z]+_?[0-9a-zA-Z]+)_?(\d)+\.qasm"
        )
        result = pattern.match(qasm_path)
        layout_name = result[2]
        idx = result[3]
        logger.info(f"{layout_name}: {circ.width()}")

        mapped_circ = map_circ(circ, layouts[layout_name])
        with open(osp.join(v_data_dir, f"mapped_{layout_name}_{idx}.qasm"), "w") as f:
            f.write(mapped_circ.qasm())


def check(layout_name: str):
    data_dir = osp.abspath(osp.dirname(__file__))
    data_dir = osp.join(data_dir, "data")
    topo_dir = osp.join(data_dir, "topo")
    layouts = {}
    for _, _, file_names in os.walk(topo_dir):
        for file_name in file_names:
            layout = Layout.load_file(osp.join(topo_dir, file_name))
            layouts[layout.name] = layout

    qasm_paths = []
    v_data_dir = osp.join(data_dir, "v_data")
    for _, _, file_names in os.walk(v_data_dir):
        for file_name in file_names:
            if not file_name.startswith("mapped") or layout_name not in file_name:
                continue
            qasm_path = osp.join(v_data_dir, file_name)
            qasm_paths.append(qasm_path)
    cnt = 0
    for qasm_path in qasm_paths:
        circ: Circuit = OPENQASMInterface.load_file(qasm_path).circuit
        cur = len(circ.gates)
        logger.info(f"{qasm_path}: {cur}")
        cnt += cur
    logger.info(f"Total #gate: {cnt}")


if __name__ == "__main__":
    # main()
    check("grid_3x3")
