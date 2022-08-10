from typing import Iterable
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import BasicGate
from os import walk, rename, makedirs
from os import path as osp
from QuICT.tools.interface import OPENQASMInterface
import re
from typing import Tuple, Iterable
import torch
from QuICT.qcda.mapping.ai.data_def import PairData
from torch_geometric.nn.models import Node2Vec
from QuICT.core.layout import Layout, LayoutEdge


class MappingDataProcessor:
    def __init__(
        self, topo_feature_dim: int, lc_feature_dim: int, working_dir: str = ".",
    ) -> None:
        self._data_dir = osp.join(working_dir, "data")
        self._circ_dir = osp.join(self._data_dir, "circ")
        self._topo_dir = osp.join(self._data_dir, "topo")
        self._processed_dir = osp.join(self._data_dir, "processed")

        self._topo_feature_dim = topo_feature_dim
        self._lc_feature_dim = lc_feature_dim

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

    def _build_circ_edge_index(self, circ: Circuit) -> torch.Tensor:
        lc_node_num = 0
        edge_index = []
        qubit_num = circ.width()
        cur_occ = [-1 for _ in range(qubit_num)]
        for gate in circ.gates:
            gate: BasicGate
            args = gate.cargs + gate.targs
            if len(args) != 2:
                continue
            # edge_index.append([lc_node_num, lc_node_num])
            for arg in args:
                if cur_occ[arg] != -1:
                    edge_index.append(
                        [cur_occ[arg], lc_node_num,]
                    )
                cur_occ[arg] = lc_node_num
            lc_node_num += 1
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return edge_index

    def _build(self) -> Iterable[Tuple[PairData, int]]:
        for topo_path, circ_path, result_circ_path in self._circ_path_list:
            layout = Layout.load_file(topo_path)
            edge_index_topo = []
            for edge in layout.edge_list:
                edge: LayoutEdge
                edge_index_topo.append((edge.u, edge.v))
                edge_index_topo.append((edge.v, edge.u))
            edge_index_topo = (
                torch.tensor(edge_index_topo, dtype=torch.long).t().contiguous()
            )
            x_topo = torch.eye(self._topo_feature_dim, dtype=torch.float)

            lc_circ: Circuit = OPENQASMInterface.load_file(circ_path).circuit
            res_circ: Circuit = OPENQASMInterface.load_file(result_circ_path).circuit

            swap_cnt = len(res_circ.gates) - len(lc_circ.gates)

            edge_index_lc = self._build_circ_edge_index(lc_circ)
            x_lc = torch.eye(self._lc_feature_dim, dtype=torch.float)
            pair_data = PairData(edge_index_topo, x_topo, edge_index_lc, x_lc)

            yield pair_data, swap_cnt

    def build(self):
        if not osp.exists(self._processed_dir):
            makedirs(self._processed_dir)
        for idx, data in enumerate(self._build()):
            f_name = osp.join(self._processed_dir, f"{idx}.pt")
            torch.save(data, f_name)
            if (idx + 1) % 100 == 0:
                print(f"    Processed {idx+1} files.")


if __name__ == "__main__":
    processor = MappingDataProcessor(topo_feature_dim=50, lc_feature_dim=1000)
    # processor.wash_circuits()
    processor.build()

