from operator import mod
from sys import flags
from typing import Iterable, List, Set, Tuple, Union
from QuICT.core import *
from QuICT.core.gate import CompositeGate, build_gate
from QuICT.core.layout.layout import LayoutEdge
from QuICT.core.utils.circuit_info import CircuitBased
from QuICT.core.gate.gate import BasicGate
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.mapping.ai.dataset import MappingHeteroDataset
from QuICT.qcda.mapping.ai.data_processor import MappingDataProcessor
from QuICT.qcda.mapping.ai.data_def import PairData
from QuICT.qcda.mapping.ai.swap_num_predict_hetero import SwapNumPredictHeteroMix
from torch_geometric.data import Batch as PygBatch
import torch
import networkx as nx
from copy import deepcopy, copy


def beam_search_mapping(
    circ: Union[Circuit, CompositeGate],
    layout: Layout,
    model: SwapNumPredictHeteroMix,
    max_qubit_num=50,
    init_mapping: Iterable[int] = None,
) -> CompositeGate:
    if init_mapping is None:
        init_mapping = [i for i in range(len(circ.gates))]

    pre_processor = MappingDataProcessor(max_qubit_num)

    # Use a closure to avoid name pollution
    def _build_base_graph() -> nx.DiGraph:
        graph = nx.DiGraph()
        for i, gate in enumerate(circ.gates):
            occupied = [-1 for _ in range(circ.width())]
            gate: BasicGate
            graph.add_node(i)
            if gate.controls + gate.targets == 2:
                for b in gate.cargs + gate.targs:
                    if occupied[b] != -1:
                        graph.add_edge(occupied[b], i)
                    occupied[b] = i
        return graph

    def _build_topo_graph(layout: Layout) -> nx.Graph:
        graph = nx.Graph()
        for edge in layout.edge_list:
            edge: LayoutEdge
            graph.add_edge(edge.u, edge.v)
        return graph

    topo_graph = _build_topo_graph(layout)
    topo_dist = dict(nx.all_pairs_shortest_path_length(topo_graph))
    topo_dist: List[List[int]] = [
        list(topo_dist[i]) for i in range(layout.qubit_number)
    ]

    # A beam contains several (physical_circuit, logical_circuit, Dep-Dag, cur_mapping, swap_count) tuples.
    beam: List[Tuple[CompositeGate, Set[int], nx.DiGraph, List[int], int]]
    beam = [
        (
            CompositeGate(),
            set([i for i in range(len(circ.gates))]),
            _build_base_graph(),
            [i for i in range(circ.width())],
            0,
        )
    ]

    edge_index_topo = pre_processor._build_topo_edge_index(layout)
    x_topo = pre_processor._build_topo_x(layout)

    def _beam_cmp_key(
        state: Tuple[CompositeGate, Set[int], nx.DiGraph, List[int], int]
    ) -> int:
        cnt = state[4]
        # TODO: add nn inference count
        cur_mapping = state[3]
        lc_node_num = 0
        edge_index_lc = []
        qubit_num = circ.width()
        cur_occ = [-1 for _ in range(qubit_num)]
        gate_labels = []
        for gate_id in state[2].nodes:
            gate = circ.gates[gate_id]
            args = gate.cargs + gate.targs
            args = [cur_mapping[arg] for arg in args]
            if len(args) != 2:
                continue
            gate_labels.append(args)
            for arg in args:
                if cur_occ[arg] != -1:
                    edge_index_lc.append(
                        [
                            cur_occ[arg],
                            lc_node_num,
                        ]
                    )
                cur_occ[arg] = lc_node_num
            lc_node_num += 1
        edge_index_lc = torch.tensor(edge_index_lc, dtype=torch.long).t().contiguous()
        x_lc = torch.zeros(lc_node_num, max_qubit_num, dtype=torch.float)
        for i, gate_label in enumerate(gate_labels):
            a, b = gate_label
            x_lc[i, a] = 1
            x_lc[i, b] = 1

        pair_data = PairData(
            edge_index_topo=edge_index_topo,
            x_topo=x_topo,
            edge_index_lc=edge_index_lc,
            x_lc=x_lc,
        )

        data = MappingHeteroDataset.to_hetero_data(pair_data)
        data = PygBatch.from_data_list([data])
        inferred = model(data)
        cnt += int(torch.squeeze(inferred))

        return cnt

    ans_circ = CompositeGate()
    ans_cnt = len(circ.gates) * circ.width() * 10
    beam_width = int(circ.width() * 1.5)

    while len(beam) > 0:
        window = len(beam)
        for pc, lc, lc_dep_graph, cur_mapping, swap_cnt in beam[:window]:
            if len(lc) == 0 and swap_cnt < ans_cnt:
                ans_circ = pc
                ans_cnt = swap_cnt
                continue
            lc_loop_cpy = copy(lc)
            for i in lc_loop_cpy:
                # Eagerly execute any single bit gate
                if circ.gates[i].controls + circ.gates[i].targets == 1:
                    print(
                        f"Physical circuit length {len(pc.gates)}. Logical circuit length {len(lc)}. Bypassing a {circ.gates[i].type}."
                    )
                    pc.append(
                        build_gate(
                            circ.gates[i].type,
                            [cur_mapping[circ.gates[i].targ]],
                            circ.gates[i].pargs,
                        )
                    )
                    lc.remove(i)
                    beam.append((pc, lc, lc_dep_graph, cur_mapping, swap_cnt))
                    continue
                # Execute by topological order
                if lc_dep_graph.in_degree(i) > 0:
                    continue

                gate: BasicGate = circ.gates[i]
                a, b = gate.cargs + gate.targs
                a, b = cur_mapping[a], cur_mapping[b]

                # Execute a 2bit gate if its satisfied by current mapping
                if topo_dist[a][b] == 1:
                    pc_cpy = deepcopy(pc)
                    lc_cpy = copy(lc)
                    lc_dep_graph_cpy = lc_dep_graph.copy()
                    cur_mapping_cpy = copy(cur_mapping)

                    pc_cpy.append(build_gate(gate.type, [a, b], gate.pargs))
                    lc_cpy.remove(i)
                    lc_dep_graph_cpy.remove_node(i)
                    beam.append(
                        (pc_cpy, lc_cpy, lc_dep_graph_cpy, cur_mapping_cpy, swap_cnt)
                    )

                # Search all possible
                potential_swap = set()
                for i in range(circ.width()):
                    if i != a:
                        potential_swap.add((i, a))
                    if i != b:
                        potential_swap.add((i, b))

                for x, y in potential_swap:
                    pc_cpy = deepcopy(pc)
                    lc_cpy = copy(lc)
                    lc_dep_graph_cpy = lc_dep_graph.copy()
                    cur_mapping_cpy = copy(cur_mapping)

                    pc_cpy.append(build_gate(GateType.swap, [x, y]))
                    cur_mapping_cpy[x], cur_mapping_cpy[y] = (
                        cur_mapping[y],
                        cur_mapping[x],
                    )
                    beam.append(
                        (
                            pc_cpy,
                            lc_cpy,
                            lc_dep_graph_cpy,
                            cur_mapping_cpy,
                            swap_cnt + 1,
                        )
                    )

        beam.sort(key=_beam_cmp_key, reverse=True)
        print("Beam augmentation stage finished.")
        print(f"Current best physical circuit length: {len(beam[0][0].gates)}.")
        print(f"Current best logical circuit length: {len(beam[0][1])}.")
        if len(beam) > beam_width:
            print("Cut off beam width.")
            beam = beam[:beam_width]

    return ans_circ


if __name__ == "__main__":
    import os.path as osp

    dataset = MappingHeteroDataset()
    model = SwapNumPredictHeteroMix(dataset[0][0].metadata(), 200, 200)
    cwd = osp.dirname(osp.abspath(__file__))
    model_path = osp.join(cwd, "model")
    model_path = osp.join(model_path, "model_1660649088_192.pt")
    model.load_state_dict(torch.load(model_path))

    layout_path = osp.join(cwd, "data")
    layout_path = osp.join(layout_path, "topo")
    layout_path = osp.join(layout_path, "ibmq_jakarta.layout")
    layout = Layout.load_file(layout_path)

    topo_graph = nx.Graph()
    for edge in layout.edge_list:
        edge: LayoutEdge
        topo_graph.add_edge(edge.u, edge.v)
    topo_dist = dict(nx.all_pairs_shortest_path_length(topo_graph))

    gate_num = 10
    circ = Circuit(layout.qubit_number)
    circ.random_append(gate_num)
    # circ.draw()

    while True:
        flag = False
        for gate in circ.gates:
            gate: BasicGate
            if gate.controls + gate.targets == 2:
                a, b = gate.cargs + gate.targs
                if topo_dist[a][b] > 1:
                    flag = True
        if flag:
            break
        print("This random circuit does not need mapping. Regenerating...")
        circ = Circuit(layout.qubit_number)
        circ.random_append(gate_num)

    print(f"Circuit info:\n1-bit gate: {circ.count_1qubit_gate()}\n2-bit gate: {circ.count_2qubit_gate()}")

    print("Start beam search")
    circ_mapped = beam_search_mapping(circ, layout, model)
