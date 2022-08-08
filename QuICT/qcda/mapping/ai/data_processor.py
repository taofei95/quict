from typing import Iterable, List, Tuple

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import BasicGate
from QuICT.core.layout import Layout, LayoutEdge
from QuICT.qcda.mapping.ai.data_def import PairData
from os import path, walk, replace
from QuICT.tools.interface import OPENQASMInterface
import torch
import re


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
    ) -> Tuple[PairData, int]:
        x_topo = torch.zeros(
            (src_circ.width(), self.topo_feature_dim), dtype=torch.float
        )
        x_topo[:, : src_circ.width()] = torch.eye(src_circ.width(), dtype=torch.float)
        edge_index_topo = []
        for edge in pc_conn:
            edge_index_topo.append((edge.u, edge.v,))
            edge_index_topo.append((edge.v, edge.u,))
        edge_index_topo = (
            torch.tensor(edge_index_topo, dtype=torch.long).t().contiguous()
        )

        lc_node_num = 0
        edge_index_lc = []
        qubit_num = src_circ.width()
        cur_occ = [-1 for _ in range(qubit_num)]
        for gate in src_circ.gates:
            gate: BasicGate
            args = gate.cargs + gate.targs
            if len(args) != 2:
                continue
            for arg in args:
                if cur_occ[arg] != -1:
                    edge_index_lc.append(
                        [cur_occ[arg], lc_node_num,]
                    )
                cur_occ[arg] = lc_node_num
            lc_node_num += 1
        x_lc = torch.zeros((lc_node_num, self.lc_feature_dim), dtype=torch.float)
        x_lc[:, :lc_node_num] = torch.eye(lc_node_num, dtype=torch.float)

        edge_index_lc = torch.tensor(edge_index_lc, dtype=torch.long).t().contiguous()

        extra_swap_cnt = len(res_circ.gates) - len(src_circ.gates)

        pair_data = PairData(
            edge_index_topo=edge_index_topo,
            x_topo=x_topo,
            edge_index_lc=edge_index_lc,
            x_lc=x_lc,
        )

        return pair_data, extra_swap_cnt

    def _load_by_topo_name(self, topo_name: str) -> Iterable[Tuple[PairData, int]]:
        layout_edges = self._load_layout_edge(topo_name)
        for src_circ, res_circ in self._load_circuits(topo_name):
            yield self._build_data_from_circ(layout_edges, src_circ, res_circ)

    def build(self):
        from os import makedirs
        from threading import Thread

        class ProcessTask(Thread):
            def __init__(self, processor: MappingDataProcessor, topo_name: str):
                Thread.__init__(self=self)
                self.topo_name = topo_name
                self.processor = processor

            def run(self):
                processed_cnt = 0
                processed_dir = self.processor.processed_dir
                for data in self.processor._load_by_topo_name(self.topo_name):
                    f_path = path.join(processed_dir, f"{self.topo_name}_{processed_cnt}.pt")
                    with open(f_path, "wb") as f:
                        torch.save(data, f)
                    processed_cnt += 1

                    if processed_cnt % 100 == 0:
                        print(
                            f"    Processed {processed_cnt} data files({self.topo_name})."
                        )
                    # break # Debug use

        if not path.exists(self.processed_dir):
            makedirs(self.processed_dir)
        print("Pre-processing all data...")
        tlist: List[ProcessTask] = []
        for name in self._get_topo_names():
            print(f"Processing circuits under {name}")
            t = ProcessTask(self, name)
            t.start()
            tlist.append(t)
        for t in tlist:
            t.join()


if __name__ == "__main__":
    processor = MappingDataProcessor(topo_feature_dim=50, lc_feature_dim=4000)
    processor.build()

