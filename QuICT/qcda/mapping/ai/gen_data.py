import os
from math import ceil, sqrt
from typing import List

from QuICT.core.circuit import *
from QuICT.core.gate import *
from QuICT.core.layout import *
from QuICT.qcda.mapping import MCTSMapping as Mapping

# from QuICT.tools.interface import OPENQASMInterface


def gen_grid_layout(num_qubit: int) -> Layout:
    # print(num_qubit)
    grid_size = int(sqrt(num_qubit))
    assert grid_size * grid_size == num_qubit

    def coo_to_idx(row: int, col: int) -> int:
        return row * grid_size + col

    layout = Layout(n=num_qubit)
    for i in range(grid_size):
        for j in range(grid_size):
            di = [0, 1]
            dj = [1, 0]
            u = coo_to_idx(i, j)
            for k in range(2):
                ni = i + di[k]
                nj = j + dj[k]
                if ni >= grid_size or nj >= grid_size:
                    continue
                v = coo_to_idx(ni, nj)
                layout.add_edge(u, v)
    return layout


def gen_fc_layout(num_qubit: int) -> Layout:
    layout = Layout(n=num_qubit)
    for i in range(num_qubit):
        for j in range(num_qubit):
            if j == i:
                continue
            layout.add_edge(i, j)
    return layout


def gen_circ(num_qubit: int) -> Circuit:
    size = num_qubit * num_qubit // 2
    # No need for single bit gates.
    rand_list = [
        GateType.cx,
        GateType.cy,
        GateType.cz,
        GateType.ch,
        GateType.crz,
        GateType.cu1,
        GateType.cu3,
        GateType.fsim,
    ]
    circ = Circuit(wires=num_qubit)
    circ.random_append(rand_size=size, typelist=rand_list)
    return circ


def ensure_path(path: str):
    dir, _ = os.path.split(os.path.abspath(path))
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def write_circ_gate_dag(circ: Circuit, path: str):
    ensure_path(path)
    num_qubit = circ.width()
    cur_occupy: List[int] = [-1 for _ in range(num_qubit)]
    edge_list = []
    for idx, gate in enumerate(circ.gates):
        gate: BasicGate
        args = gate.cargs + gate.targs
        i = cur_occupy[args[0]]
        j = cur_occupy[args[1]]
        if i >= 0:
            edge_list.append((i, idx))
        if j >= 0:
            edge_list.append((j, idx))
        cur_occupy[args[0]] = idx
        cur_occupy[args[1]] = idx
    with open(path, "w") as f:
        f.write(f"# number of qubit: {num_qubit}\n")
        for edge in edge_list:
            f.write(f"{edge[0]} {edge[1]}\n")


def write_layout(layout: Layout, path: str):
    ensure_path(path)
    # layout.write_file(path)
    with open(path, "w") as f:
        for edge in layout.edge_list:
            edge: LayoutEdge
            f.write(f"{edge.u} {edge.v}\n")


def gen_data(repeat: int = 100, size_list: List[int] = [i for i in range(5, 150, 5)]):
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "data")
    topo_list = [(gen_grid_layout, "grid"), (gen_fc_layout, "fc")]
    for topo_gen, name in topo_list:
        topo_dir = os.path.join(data_dir, name)
        for num_qubit in size_list:
            if name == "grid" and int(sqrt(num_qubit)) ** 2 != num_qubit:
                continue
            print(f"Circuit with {num_qubit} of {name} topology, {repeat} rounds.")
            dir = os.path.join(topo_dir, str(num_qubit))
            topo_file = os.path.join(dir, "topo.txt")
            topo = topo_gen(num_qubit)
            write_layout(topo, topo_file)
            for i in range(repeat):
                circ_file = os.path.join(dir, f"circ_{i}.txt")
                circ = gen_circ(num_qubit)
                write_circ_gate_dag(circ, circ_file)


if __name__ == "__main__":
    gen_data()
