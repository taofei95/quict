import re
from os import makedirs
from os import path as osp
from os import rename, walk
from typing import Any, Iterable, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.core.gate.gate import BasicGate
from QuICT.core.layout import Layout, LayoutEdge
from QuICT.core.utils.circuit_info import CircuitBased
from QuICT.qcda.mapping.ai.data_def import PairData
from QuICT.tools.interface import OPENQASMInterface
from torch_geometric.data import HeteroData
from torch_geometric.nn.models import Node2Vec


class MappingDataProcessor:
    def __init__(
        self,
        max_qubit_num: int,
        working_dir: str = None,
    ) -> None:
        if working_dir is None:
            working_dir = osp.dirname(osp.abspath(__file__))

        self._data_dir = osp.join(working_dir, "data")
        self._circ_dir = osp.join(self._data_dir, "circ")
        self._topo_dir = osp.join(self._data_dir, "topo")
        self._processed_dir = osp.join(self._data_dir, "processed")

        self._max_qubit_num = max_qubit_num

        self._circ_path_list = list(self._load_circuits_path())

    def _get_topo_names(self) -> Iterable[str]:
        for _, _, filenames in walk(self._topo_dir):
            for name in filenames:
                yield name.split(".")[0]

    def _wash_circuits(self, topo_name):
        cnt = 0
        for root, _, filenames in walk(osp.join(self._circ_dir, topo_name)):
            for name in filenames:
                if name.startswith("result") or name.startswith("skip"):
                    continue
                cnt += 1
                if cnt % 100 == 0:
                    print(f"    Processed {cnt} files.")

                circ_path = osp.join(root, name)

                circ: Circuit = OPENQASMInterface.load_file(circ_path).circuit

                no_2bit_gate = True
                i = 0
                while no_2bit_gate and i < len(circ.gates):
                    g: BasicGate = circ.gates[i]
                    n = len(g.cargs + g.targs)
                    if n == 2:
                        no_2bit_gate = False
                    i += 1
                if no_2bit_gate:
                    # Rename the raw qasm to skip them during training
                    id = re.search(r"\d+", circ_path).group(0)
                    replace_name = circ_path.replace(f"circ_{id}", f"skip_circ{id}")
                    print(
                        f"There's no 2 bit gate in {circ_path}. Rename it as {replace_name}."
                    )
                    rename(circ_path, replace_name)

    def wash_circuits(self):
        for topo_name in self._get_topo_names():
            print(f"Processing circuits under {topo_name}...")
            self._wash_circuits(topo_name)

    def _get_topo_names(self) -> Iterable[str]:
        for _, _, filenames in walk(self._topo_dir):
            for name in filenames:
                yield name.split(".")[0]

    def _load_circuits_path(self) -> Iterable[Tuple[str, str, str]]:
        for topo_name in self._get_topo_names():
            topo_path = osp.join(self._topo_dir, f"{topo_name}.layout")
            for root, _, filenames in walk(osp.join(self._circ_dir, topo_name)):
                for name in filenames:
                    if name.startswith("result") or name.startswith("skip"):
                        continue
                    circ_path = osp.join(root, name)
                    result_circ_path = osp.join(root, f"result_{name}")
                    yield topo_path, circ_path, result_circ_path

    def _build_circ_edge_index(
        self, circ: CircuitBased
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lc_node_num = 0
        edge_index = []
        qubit_num = circ.width()
        cur_occ = [-1 for _ in range(qubit_num)]
        gate_labels = []
        for gate in circ.gates:
            gate: BasicGate
            args = gate.cargs + gate.targs
            if len(args) != 2:
                continue
            gate_labels.append(args)
            for arg in args:
                if cur_occ[arg] != -1:
                    edge_index.append(
                        [
                            cur_occ[arg],
                            lc_node_num,
                        ]
                    )
                cur_occ[arg] = lc_node_num
            lc_node_num += 1
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        x = torch.zeros(lc_node_num, self._max_qubit_num, dtype=torch.float)
        for i, gate_label in enumerate(gate_labels):
            a, b = gate_label
            x[i, a] = 1
            x[i, b] = 1
        return edge_index, x

    def _build_topo_edge_index(self, layout: Layout) -> torch.Tensor:
        edge_index_topo = []
        for edge in layout.edge_list:
            edge: LayoutEdge
            edge_index_topo.append((edge.u, edge.v))
            edge_index_topo.append((edge.v, edge.u))
        return torch.tensor(edge_index_topo, dtype=torch.long).t().contiguous()

    def _build_topo_x(self, layout: Layout) -> torch.Tensor:
        x_topo = torch.zeros(
            self._max_qubit_num, self._max_qubit_num * 2, dtype=torch.float
        )
        x_topo[:, : self._max_qubit_num] = torch.eye(
            self._max_qubit_num, dtype=torch.float
        )
        graph = nx.Graph()
        for edge in layout.edge_list:
            edge: LayoutEdge
            graph.add_edge(edge.u, edge.v)
        shortest = dict(nx.all_pairs_shortest_path_length(graph))
        nq = layout.qubit_number
        x_topo[:nq, self._max_qubit_num : nq + self._max_qubit_num] = torch.tensor(
            [list(shortest[i]) for i in range(nq)], dtype=float
        )
        return x_topo

    def _build_pair_data(self, lc_circ: CircuitBased, layout: Layout) -> PairData:
        edge_index_topo = self._build_topo_edge_index(layout)
        x_topo = self._build_topo_x(layout)
        edge_index_lc, x_lc = self._build_circ_edge_index(lc_circ)
        return PairData(edge_index_topo, x_topo, edge_index_lc, x_lc)

    def _build_layered_data(
        self, lc_circ: CircuitBased, layout: Layout
    ) -> List[HeteroData]:
        """Build circuit representation by layers.

        Args:
            lc_circ (CircuitBased): Input circuit.
            layout (Layout): Input layout.

        Returns:
            List[HeteroData]: A list of (topo, logical circuit) heterogenous data.
        """
        max_qubit_number = self._max_qubit_num
        qubit_number = layout.qubit_number
        layers_raw: List[List[Tuple[int, int]]] = []
        occupied = [-1 for _ in range(max_qubit_number)]
        for gate in lc_circ.gates:
            gate: BasicGate
            if gate.controls + gate.targets != 2:
                continue
            a, b = gate.cargs + gate.targs
            idx = max(occupied[a], occupied[b]) + 1
            while idx >= len(layers_raw):
                layers_raw.append([])
            layers_raw[idx].append((a, b))
            layers_raw[idx].append((b, a))
            occupied[a] = idx
            occupied[b] = idx
        lc_edge_index_layers = []
        for layer in layers_raw:
            lc_edge_index_layers.append(
                torch.tensor(layer, dtype=torch.long).T.contiguous()
            )
        lc_x = torch.zeros(max_qubit_number, max_qubit_number, dtype=torch.float)
        lc_x[:qubit_number, :qubit_number] = torch.eye(qubit_number, dtype=torch.float)

        topo_edge_index = []
        for edge in layout.edge_list:
            edge: LayoutEdge
            topo_edge_index.append([edge.u, edge.v])
            topo_edge_index.append([edge.v, edge.u])
        topo_edge_index = torch.tensor(topo_edge_index, dtype=torch.long).T.contiguous()
        topo_x = lc_x.clone()

        layers = []

        for lc_edge_index_layer in lc_edge_index_layers:
            data = HeteroData()
            data["topo"].x = topo_x.clone()
            data["lc"].x = lc_x.clone()

            data["topo", "topo_edge", "topo"].edge_index = topo_edge_index.clone()
            data["lc", "lc_edge", "lc"].edge_index = lc_edge_index_layer

            layers.append(data)

        return layers

    def _build_layered_data_all(self):
        for topo_path, circ_path, result_circ_path in self._circ_path_list:
            layout = Layout.load_file(topo_path)

            lc_circ: Circuit = OPENQASMInterface.load_file(circ_path).circuit
            res_circ: Circuit = OPENQASMInterface.load_file(result_circ_path).circuit

            swap_cnt = len(res_circ.gates) - len(lc_circ.gates)

            layer_data = self._build_layered_data(lc_circ, layout)

            yield layer_data, swap_cnt

    def _build(self) -> Iterable[Tuple[PairData, int]]:
        for topo_path, circ_path, result_circ_path in self._circ_path_list:
            layout = Layout.load_file(topo_path)

            lc_circ: Circuit = OPENQASMInterface.load_file(circ_path).circuit
            res_circ: Circuit = OPENQASMInterface.load_file(result_circ_path).circuit

            swap_cnt = len(res_circ.gates) - len(lc_circ.gates)

            pair_data = self._build_pair_data(lc_circ, layout)

            yield pair_data, swap_cnt

    def build(self, data_generator: Any = None, process_dir: str = None):
        if process_dir is None:
            process_dir = self._processed_dir

        if data_generator is None:
            data_generator = self._build()

        print(f"Save data into {process_dir}")

        if not osp.exists(process_dir):
            makedirs(process_dir)
        for idx, data in enumerate(data_generator):
            f_name = osp.join(process_dir, f"{idx}.pt")
            torch.save(data, f_name)
            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx+1} files.")


if __name__ == "__main__":
    processor = MappingDataProcessor(max_qubit_num=50)
    # processor.wash_circuits()
    process_dir = osp.dirname(osp.abspath(__file__))
    process_dir = osp.join(process_dir, "data")
    process_dir = osp.join(process_dir, "processed_layered")
    processor.build(
        data_generator=processor._build_layered_data_all(), process_dir=process_dir
    )
