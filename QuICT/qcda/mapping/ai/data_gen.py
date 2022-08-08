from ast import arg
import os
from math import ceil, sqrt
from typing import List
from random import randint
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
    size = randint(10, 250)
    # No need for single bit gates.
    rand_list = [
        GateType.cx,
        GateType.rx,
        GateType.ry,
    ]
    circ = Circuit(wires=num_qubit)
    circ.random_append(rand_size=size, typelist=rand_list)
    return circ


def ensure_path(path: str):
    if not os.path.exists(path):
        print(f"Creating absent {path}...")
        os.makedirs(path, exist_ok=True)


def write_circ_qasm(circ: Circuit, path: str):
    with open(path, "w") as f:
        f.write(circ.qasm())


def gen_grid_data(repeat: int = 100, size_list: List[int] = [i for i in range(5, 50)]):
    cwd = os.getcwd()
    data_dir = os.path.join(cwd, "data")
    topo_list = [(gen_grid_layout, "grid")]
    for topo_gen, name in topo_list:
        topo_dir = os.path.join(data_dir, name)
        for num_qubit in size_list:
            if name == "grid" and int(sqrt(num_qubit)) ** 2 != num_qubit:
                continue
            print(
                f"Circuit with {num_qubit} qubits of {name} topology, {repeat} rounds."
            )
            dir = os.path.join(topo_dir, str(num_qubit))
            # topo_file = os.path.join(dir, "topo.txt")
            topo = topo_gen(num_qubit)
            # write_layout(name, num_qubit, topo, topo_file)
            for i in range(repeat):
                circ_file = os.path.join(dir, f"circ_{i}.qasm")
                circ = gen_circ(num_qubit)
                result_circ_file = os.path.join(dir, f"result_circ_{i}.qasm")
                result_circ = Mapping.execute(
                    circuit=circ, init_mapping_method="naive", layout=topo
                )
                # write_circ_gate_dag(circ, circ_file)
                write_circ_qasm(circ, circ_file)
                write_circ_qasm(result_circ, result_circ_file)
                if 0 == i % 10:
                    print(f"    iter: {i}")


def get_ibmq_topo():
    for root, _, files in os.walk("data"):
        for name in files:
            if name.startswith("ibmq"):
                path = os.path.join(root, name)
                exists = os.path.isfile(path)
                # print(f"{path}, {exists}")
                if exists:
                    yield path


def task_seg(circ_dir: str, layout: Layout, repeat: int, begin: int, end):
    for i in range(begin, end):
        if 0 == i % 10:
            print(f"    iter: {i}")
        circ = gen_circ(layout.qubit_number)
        circ_file = os.path.join(circ_dir, f"circ_{i}.qasm")
        result_circ_file = os.path.join(circ_dir, f"result_circ_{i}.qasm")
        result_circ = Mapping.execute(
            circuit=circ, init_mapping_method="naive", layout=layout, num_of_process=8
        )
        write_circ_qasm(circ, circ_file)
        write_circ_qasm(result_circ, result_circ_file)
        del circ
        del circ_file


def run(x):
    return x[0](*x[1:])


def gen_ibmq_data(epoch: int, repeat: int):
    for topo_path in get_ibmq_topo():
        topo_name = os.path.basename(topo_path)
        topo_name = os.path.splitext(topo_name)[0]

        layout = Layout.load_file(topo_path)
        print(f"[Epoch {epoch}] Generating {topo_name} for {repeat} rounds...")
        circ_dir = os.path.join("data", "circ")
        circ_dir = os.path.join(circ_dir, topo_name)
        print(circ_dir)
        ensure_path(circ_dir)

        task_seg(
            circ_dir=circ_dir,
            layout=layout,
            repeat=repeat,
            begin=epoch * repeat,
            end=epoch * repeat + repeat,
        )


if __name__ == "__main__":
    from sys import argv

    epoch = 1
    if len(argv) > 1:
        try:
            epoch = int(argv[1])
        except:
            pass
    print(f"Generating data using {epoch} epochs...")
    gen_ibmq_data(epoch=epoch, repeat=200)
