#!/usr/bin/env python3

import os
from os import path as osp
from typing import Dict, List

import numpy as np

from QuICT.core import *
from QuICT.core.gate.gate import BasicGate
from QuICT.core.utils.gate_type import GateType
from QuICT.simulation.state_vector import CircuitSimulator

data: Dict[str, list] = {}

data["1bit-all"] = [
    GateType.x,
    GateType.y,
    GateType.z,
    GateType.h,
    GateType.rx,
    GateType.ry,
    GateType.rz,
    GateType.u1,
    GateType.u2,
    GateType.u3,
    GateType.s,
    GateType.sdg,
]

data["2bit-all"] = [
    GateType.rxx,
    GateType.ryy,
    GateType.rzz,
    GateType.fsim,
    GateType.cu1,
    GateType.cu3,
    GateType.swap,
    GateType.cx,
    GateType.cy,
    GateType.cz,
    GateType.ch,
]


def gen_desc(category: str, circ: Circuit) -> List[str]:
    desc = [f"qubit: {circ.width()}\n"]
    if "diag" in category:
        pass
    elif "ctrl" in category:
        pass
    else:
        for gate in circ.gates:
            gate: BasicGate
            line = "tag: untary; data: "
            for e in gate.matrix.flatten():
                data_str = str(e).strip()
                if not data_str[0] == "(":
                    data_str = f"({data_str})"
                line += data_str
                line += " "
            desc.append(f"{line}\n")
    return desc


def gen_vec(circ: Circuit) -> List[str]:
    simulator = CircuitSimulator()
    amp: np.ndarray = simulator.run(circ)
    vec = []
    for e in amp.flatten():
        line = str(e)
        if not line[0] == "(":
            line = f"({line})"
        vec.append(f"{line}\n")
    return vec


def main():
    script_path = osp.dirname(osp.abspath(__file__))
    data_path = osp.join(script_path, "data")
    if not osp.exists(data_path):
        os.makedirs(data_path)
    for category, lst in data.items():
        for q in range(2, 8):
            for i in range(10):
                size = (i + 1) * 20
                tag = f"{category}_qubit{q}_size{size}"
                desc_f_name = osp.join(data_path, f"desc_{tag}")
                vec_f_name = osp.join(data_path, f"vec_{tag}")
                circ = Circuit(q)
                circ.random_append(rand_size=size, typelist=lst, random_params=True)

                desc = gen_desc(category, circ)
                with open(desc_f_name, "w", encoding="utf-8") as f:
                    f.writelines(desc)

                vec = gen_vec(circ)
                with open(vec_f_name, "w", encoding="utf-8") as f:
                    f.writelines(vec)


if __name__ == "__main__":
    main()
