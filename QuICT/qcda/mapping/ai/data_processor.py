from ctypes.wintypes import LPSC_HANDLE
from typing import Dict, Generator, Iterable, List, Tuple, Union

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import BasicGate
from QuICT.core.layout import Layout, LayoutEdge
from os import path, walk
from QuICT.tools.interface import OPENQASMInterface
import torch
from torch_geometric.data import HeteroData, Data
import torch_geometric.transforms as GT
import numpy as np
import networkx as nx


class MappingDataProcessor:
    def __init__(self, working_dir: str = ".",) -> None:
        self._data_dir = path.join(working_dir, "data")
        self._circ_dir = path.join(self._data_dir, "circ")
        self._topo_dir = path.join(self._data_dir, "topo")
        self._processed_dir = path.join(self._data_dir, "processed")

    def _get_topo_names(self) -> Iterable[str]:
        for _, _, filenames in walk(self._topo_dir):
            for name in filenames:
                yield name.split(".")[0]

    def _load_layout_edge(self, topo_name) -> List[LayoutEdge]:
        layout = Layout.load_file(path.join(self._topo_dir, f"{topo_name}.layout"))
        return layout.edge_list

    def _load_circuits(self, topo_name) -> Iterable[Tuple[Circuit, Circuit]]:
        for root, _, filenames in walk(path.join(self._circ_dir, topo_name)):
            for name in filenames:
                if name.startswith("result"):
                    continue
                circ_path = path.join(root, name)
                result_circ_path = path.join(root, f"result_{name}")
                # print(circ_path)
                circuit: Circuit = OPENQASMInterface.load_file(circ_path).circuit
                result_circuit: Circuit = OPENQASMInterface.load_file(
                    result_circ_path
                ).circuit
                yield circuit, result_circuit

    def _build_data_from_circ(
        self, pc_conn: List[LayoutEdge], src_circ: Circuit, res_circ: Circuit
    ) -> Tuple[Dict[str, Data], int]:
        topo_x = torch.eye(src_circ.width(), dtype=torch.long)
        topo_edge_index = []
        for edge in pc_conn:
            topo_edge_index.append((edge.u, edge.v,))
            topo_edge_index.append((edge.v, edge.u,))
        topo_edge_index = (
            torch.tensor(topo_edge_index, dtype=torch.float).t().contiguous()
        )
        topo_data = Data(x=topo_x, edge_index=topo_edge_index)

        dag = nx.DiGraph()  # Load from circuit in the future
        lc_node_num = -1
        lc_x = torch.eye(lc_node_num, dtype=torch.long)
        lc_edge_index = []
        for node in nx.topological_sort(dag):
            for neighbor in dag[node]:
                lc_edge_index.append((node, neighbor,))
        lc_edge_index = torch.tensor(lc_edge_index, dtype=torch.float).t().contiguous()
        lc_data = Data(x=lc_x, edge_index=lc_edge_index)

        extra_swap_cnt = len(res_circ.gates) - len(src_circ.gates)

        data_dict = {}
        data_dict["topo"] = topo_data
        data_dict["lc"] = lc_data

        return data_dict, extra_swap_cnt

    def _load_by_topo_name(
        self, topo_name: str
    ) -> Iterable[Tuple[Dict[str, Data], int]]:
        layout_edges = self._load_layout_edge(topo_name)
        for src_circ, res_circ in self._load_circuits(topo_name):
            yield self._build_data_from_circ(layout_edges, src_circ, res_circ)

    def build(self):
        from os import makedirs

        processed_cnt = 0
        if not path.exists(self._processed_dir):
            makedirs(self._processed_dir)
        print("Pre-processing all data...")
        for name in self._get_topo_names():
            print(f"    Processing circuits under {name}")
            for data in self._load_by_topo_name(name):
                f_path = path.join(self._processed_dir, f"{processed_cnt}.pt")
                with open(f_path, "wb") as f:
                    torch.save(data, f)
                processed_cnt += 1

                if processed_cnt % 100 == 0:
                    print(f"        Processed {processed_cnt} data files.")


if __name__ == "__main__":
    processor = MappingDataProcessor()
    processor.build()

