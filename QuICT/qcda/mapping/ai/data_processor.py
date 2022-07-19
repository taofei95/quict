from typing import Dict, Iterable, List, Tuple

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
    def __init__(
        self, topo_feature_dim: int, lc_feature_dim: int, working_dir: str = ".",
    ) -> None:
        self.data_dir = path.join(working_dir, "data")
        self.circ_dir = path.join(self.data_dir, "circ")
        self.topo_dir = path.join(self.data_dir, "topo")
        self.processed_dir = path.join(self.data_dir, "processed")
        self.topo_feature_dim = topo_feature_dim
        self.lc_feature_dim = lc_feature_dim

    def _get_topo_names(self) -> Iterable[str]:
        for _, _, filenames in walk(self.topo_dir):
            for name in filenames:
                yield name.split(".")[0]

    def _load_layout_edge(self, topo_name) -> List[LayoutEdge]:
        layout = Layout.load_file(path.join(self.topo_dir, f"{topo_name}.layout"))
        return layout.edge_list

    def _load_circuits(self, topo_name) -> Iterable[Tuple[Circuit, Circuit]]:
        for root, _, filenames in walk(path.join(self.circ_dir, topo_name)):
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
        topo_x = torch.zeros(
            (src_circ.width(), self.topo_feature_dim), dtype=torch.float
        )
        topo_x[:, : src_circ.width()] = torch.eye(src_circ.width(), dtype=torch.float)
        topo_edge_index = []
        for edge in pc_conn:
            topo_edge_index.append((edge.u, edge.v,))
            topo_edge_index.append((edge.v, edge.u,))
        topo_edge_index = (
            torch.tensor(topo_edge_index, dtype=torch.long).t().contiguous()
        )
        topo_data = Data(x=topo_x, edge_index=topo_edge_index)

        lc_node_num = 0
        lc_edge_index = []
        qubit_num = src_circ.width()
        cur_occ = [-1 for _ in range(qubit_num)]
        for gate in src_circ.gates:
            gate: BasicGate
            args = gate.cargs + gate.targs
            if len(args) != 2:
                continue
            for arg in args:
                if cur_occ[arg] != -1:
                    lc_edge_index.append(
                        [cur_occ[arg], lc_node_num,]
                    )
                cur_occ[arg] = lc_node_num
            lc_node_num += 1
        lc_x = torch.zeros((lc_node_num, self.lc_feature_dim), dtype=torch.float)
        lc_x[:, :lc_node_num] = torch.eye(lc_node_num, dtype=torch.float)

        lc_edge_index = torch.tensor(lc_edge_index, dtype=torch.long).t().contiguous()
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
        if not path.exists(self.processed_dir):
            makedirs(self.processed_dir)
        print("Pre-processing all data...")
        for name in self._get_topo_names():
            print(f"    Processing circuits under {name}")
            for data in self._load_by_topo_name(name):
                f_path = path.join(self.processed_dir, f"{processed_cnt}.pt")
                with open(f_path, "wb") as f:
                    torch.save(data, f)
                processed_cnt += 1

                if processed_cnt % 100 == 0:
                    print(f"        Processed {processed_cnt} data files.")
                # break # Debug use


if __name__ == "__main__":
    processor = MappingDataProcessor(topo_feature_dim=50, lc_feature_dim=2000)
    processor.build()

