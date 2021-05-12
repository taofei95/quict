from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import torch

from QuICT.core.exception import *
from QuICT.core.gate.gate import *
from .coupling_graph import CouplingGraph
from .exception import *


def is_two_qubit_gate_equal(s1: List[int], s2: List[int]) -> bool:
    """
    Indicate whether two two-qubit gates is same
    """
    if (s1[0] == s2[0] and s1[1] == s2[1]) or (s1[0] == s2[1] and s1[1] == s2[0]):
        return True
    else:
        return False


def f(x: np.ndarray) -> np.ndarray:
    return np.piecewise(x, [x < 0, x == 0, x > 0], [0, 0.001, lambda x: x])


def transform_batch(batch_data: Tuple[np.ndarray, np.ndarray, np.ndarray], device: torch.device) -> Tuple[
    torch.LongTensor, torch.FloatTensor]:
    qubits_list, padding_mask_list, adj_list = batch_data
    padding_mask = torch.from_numpy(padding_mask_list).to(torch.uint8)
    qubits = torch.from_numpy(qubits_list).long()

    adj_list = get_adj_list(adj_list)
    if isinstance(adj_list, list):
        adj = torch.from_numpy(np.concatenate(adj_list, axis=0)).long()
    else:
        adj = torch.from_numpy(adj_list).long()

    return qubits.to(device), padding_mask.to(device), adj.to(device)


def get_adj_list(adj_list: List[np.ndarray]) -> np.ndarray:
    if isinstance(adj_list, list):
        idx_bias = 0
        for i in range(len(adj_list)):
            adj_list[i] = adj_list[i] + idx_bias
            idx_bias += adj_list[i].shape[0]
    return adj_list


def get_graph_pool(batch_graph: List[np.ndarray], device: torch.device) -> torch.FloatTensor:
    """
    """
    if isinstance(batch_graph, list):
        num_of_graph = len(batch_graph)
        num_of_nodes = 0
        bias = [0]
        for i in range(num_of_graph):
            num_of_nodes += batch_graph[i].shape[0]
            bias.append(num_of_nodes)

        elem = torch.ones(num_of_nodes, dtype=torch.float)
        idx = torch.zeros([2, num_of_nodes], dtype=torch.long)
        for i in range(num_of_graph):
            v = torch.arange(start=0, end=batch_graph[i].shape[0], dtype=torch.long)
            idx[0, bias[i]: bias[i + 1]] = i
            idx[1, bias[i]: bias[i + 1]] = v
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([num_of_graph, num_of_nodes])).to(device)

    else:
        elem = torch.ones(batch_graph.shape[0], dtype=torch.float)
        idx = torch.zeros([2, batch_graph.shape[0]], dtype=torch.long)
        idx[0, :] = 0
        idx[1, :] = torch.arange(start=0, end=batch_graph.shape[0], dtype=torch.long)
        graph_pool = torch.sparse.FloatTensor(idx, elem, torch.Size([1, batch_graph.shape[0]])).to(device)

    return graph_pool


class EdgeProb:
    def __init__(self, circuit: DAG = None, coupling_graph: CouplingGraph = None, qubit_mapping: List[int] = None,
                 gates: List[int] = None, num_of_qubits: int = 0):
        self._circuit = circuit
        self._coupling_graph = coupling_graph
        self._qubit_mapping = qubit_mapping
        self._gates = gates
        self._inverse_qubit_mapping = [-1 for _ in range(max(len(qubit_mapping), num_of_qubits))]
        for i, elm in enumerate(qubit_mapping):
            self._inverse_qubit_mapping[elm] = i

    def __call__(self, swap_index: int = -1) -> int:

        if swap_index == -1:
            return self._neareast_neighbour_count(self._gates, self._qubit_mapping)

        swap_gate = self._coupling_graph.get_swap_gate(swap_index)
        if isinstance(swap_gate, SwapGate) is not True:
            raise TypeException("swap gate", "other gate")
        qubit_mapping = self._change_mapping_with_single_swap(swap_gate)

        return self._neareast_neighbour_count(self._gates, qubit_mapping)

    def _neareast_neighbour_count(self, gates: List[int], cur_mapping: List[int]) -> int:
        """
        Caculate the sum of the distance of all the gates in the front layer on the physical device
        """
        NNC = 0
        for gate in gates:
            NNC = NNC + self._get_logical_gate_distance_in_device(cur_mapping, gate)
        return NNC

    def _get_logical_gate_distance_in_device(self, cur_mapping: List[int], gate: int) -> int:
        """
        return the distance between the control qubit and target qubit of the given gate  on the physical device 
        """
        if self._circuit[gate]['gate'].type() == GATE_ID['Swap']:
            qubits = self._circuit[gate]['gate'].targs
        else:
            qubits = [self._circuit[gate]['gate'].carg, self._circuit[gate]['gate'].targ]

        return self._coupling_graph.distance(cur_mapping[qubits[0]], cur_mapping[qubits[1]])

    def _change_mapping_with_single_swap(self, swap_gate: SwapGate) -> List[int]:
        """
        Get the new mapping changed by a single swap, e.g., the mapping [0, 1, 2, 3, 4] of 5 qubits will be changed
        to the mapping [1, 0, 2, 3, 4] by the swap gate SWAP(0,1).
        """
        res_mapping = self._qubit_mapping.copy()
        if isinstance(swap_gate, SwapGate):
            p_target = swap_gate.targs
            l_target = [self._inverse_qubit_mapping[i] for i in p_target]
            if self._coupling_graph.is_adjacent(p_target[0], p_target[1]):
                if not l_target[0] == -1:
                    res_mapping[l_target[0]] = p_target[1]
                if not l_target[1] == -1:
                    res_mapping[l_target[1]] = p_target[0]
            else:
                raise MappingLayoutException()
        else:
            raise TypeException("swap gate", "other gate")
        return res_mapping


@dataclass
class GNNConfig(object):
    num_of_gates: int = 150
    maximum_capacity: int = 100000
    num_of_nodes: int = 150
    maximum_circuit: int = 5000
    minimum_circuit: int = 50
    gamma: int = 0.7
    num_of_playout: int = 500
    virtual_loss: float = 1
    diri_alpha: float = 0.03
    selection_times: int = 40
    epsilon: float = 0.25
    sim_method: int = 0
    batch_size: int = 10
    ff_hidden_size: int = 128
    num_self_att_layers: int = 4
    dropout: float = 0.5
    num_U2GNN_layers: int = 1
    value_head_size: int = 256
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_name: str = 'ibmq20'
    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    num_of_epochs: int = 50
    num_of_process: int = 16
    feature_update: bool = False
    loss_c: float = 10
    mcts_c: float = 20
    extended: bool = False
    num_of_swap_gates: int = 15
    gat: bool = False
    d_embed: int = 64
    d_model: int = 64
    n_head: int = 4
    n_layer: int = 4
    n_encoder: int = 8
    n_gat: int = 2
    hid_dim: int = 128
    dim_feedforward: int = 128


default_gnn_config = GNNConfig(maximum_capacity=200000, num_of_gates=150, maximum_circuit=1500, minimum_circuit=200,
                               batch_size=256,
                               ff_hidden_size=128, num_self_att_layers=4, dropout=0.5, value_head_size=128, gamma=0.7,
                               num_U2GNN_layers=2, learning_rate=0.001, weight_decay=1e-4, num_of_epochs=1000,
                               device=torch.device("cuda"),
                               graph_name='ibmq20', num_of_process=10, feature_update=True, gat=True, n_gat=1,
                               mcts_c=20, loss_c=10,
                               num_of_swap_gates=15, sim_method=2)


@dataclass
class SharedMemoryName:
    adj_name: str
    label_name: str
    value_name: str
    qubits_name: str
    num_name: str
    action_probability_name: str


class Mode(Enum):
    """
    WHOLE_CIRCUIT: Deal with all the gates in the circuit.
    TWO_QUBIT_CIRCUIT: Only deal with the two-qubit gates in the circuit.
    """
    WHOLE_CIRCUIT = 1
    TWO_QUBIT_CIRCUIT = 2


class SimMode(Enum):
    """
    """
    AVERAGE = 0
    MAX = 1


class MCTSMode(Enum):
    """
    """
    TRAIN = 0
    EVALUATE = 1
    SEARCH = 2
    EXTENDED_PROB = 3


class RLMode(Enum):
    """
    """
    WARMUP = 0
    SELFPALY = 1


class EvaluateMode(Enum):
    SEARCH = 0
    EXTENDED_PROB = 1
    PROB = 2


class Benchmark(Enum):
    REVLIB = 0
    RANDOM = 1


class MCTSMethod(Enum):
    RL = 0
    RL_ORACLE = 1


small_benchmark = ("rd84_142",
                   "adr4_197",
                   "radd_250",
                   "z4_268",
                   "sym6_145",
                   "misex1_241",
                   "rd73_252",
                   "cycle10_2_110",
                   "square_root_7",
                   "sqn_258",
                   "rd84_253")


@dataclass
class RLConfig(object):
    num_of_gates: int = 150
    maximum_capacity: int = 100000
    maximum_circuit: int = 5000
    minimum_circuit: int = 50
    gamma: int = 0.7
    num_of_playout: int = 500
    virtual_loss: float = 1
    diri_alpha: float = 0.03
    selection_times: int = 40
    epsilon: float = 0.25
    with_predictor: bool = False
    batch_size: int = 10
    dropout: float = 0.5
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph_name: str = 'ibmq20'

    learning_rate: float = 0.1
    weight_decay: float = 1e-4
    num_of_epochs: int = 50
    num_of_process: int = 4
    num_of_circuit_process: int = 8

    loss_c: float = 10
    mcts_c: float = 20

    extended: bool = False
    num_of_swap_gates: int = 15

    gat: bool = False
    d_embed: int = 64
    d_model: int = 64
    n_head: int = 4
    n_layer: int = 4
    n_encoder: int = 8
    n_gat: int = 2
    hid_dim: int = 128
    dim_feedforward: int = 128


default_rl_config = RLConfig(num_of_gates=150,
                             maximum_capacity=100000,
                             maximum_circuit=5000,
                             minimum_circuit=50,
                             gamma=0.8,
                             num_of_playout=2,
                             virtual_loss=1,
                             diri_alpha=0.03,
                             selection_times=400,
                             epsilon=0.25,
                             batch_size=10,
                             dropout=0.5,
                             device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                             graph_name='ibmq20',
                             learning_rate=0.1,
                             weight_decay=1e-4,
                             num_of_epochs=50,
                             num_of_process=4,
                             num_of_circuit_process=8,
                             loss_c=10,
                             mcts_c=20,
                             extended=False,
                             num_of_swap_gates=15,
                             gat=False,
                             d_embed=32,
                             d_model=32,
                             n_head=2,
                             n_layer=2,
                             n_encoder=4,
                             n_gat=1,
                             hid_dim=128,
                             dim_feedforward=128,
                             with_predictor=False)

bench_mark = {
    "mcts_small": (
        "rd84_142.qasm",
        "adr4_197.qasm",
        "radd_250.qasm",
        "z4_268.qasm",
        "sym6_145.qasm",
        "misex1_241.qasm",
        "rd73_252.qasm",
        "cycle10_2_110.qasm",
        "square_root_7.qasm",
        "sqn_258.qasm",
        "rd84_253.qasm",
    ),
    "extended": (
        'graycode6_47.qasm',
        'xor5_254.qasm',
        'ex1_226.qasm',
        '4gt11_84.qasm',
        'ex-1_166.qasm',
        'ham3_102.qasm',
        '4mod5-v0_20.qasm',
        '4mod5-v1_22.qasm',
        'mod5d1_63.qasm',
        '4gt11_83.qasm',
        '4gt11_82.qasm',
        'rd32-v0_66.qasm',
        'mod5mils_65.qasm',
        '4mod5-v0_19.qasm',
        'rd32-v1_68.qasm',
        'alu-v0_27.qasm',
        '3_17_13.qasm',
        '4mod5-v1_24.qasm',
        'alu-v1_29.qasm',
        'alu-v1_28.qasm',
        'alu-v3_35.qasm',
        'alu-v2_33.qasm',
        'alu-v4_37.qasm',
        'miller_11.qasm',
        'decod24-v0_38.qasm',
        'alu-v3_34.qasm',
        'decod24-v2_43.qasm',
        'mod5d2_64.qasm',
        '4gt13_92.qasm',
        '4gt13-v1_93.qasm',
        'one-two-three-v2_100.qasm',
        '4mod5-v1_23.qasm',
        '4mod5-v0_18.qasm',
        'one-two-three-v3_101.qasm',
        '4mod5-bdd_287.qasm',
        'decod24-bdd_294.qasm',
        '4gt5_75.qasm',
        'alu-v0_26.qasm',
        'rd32_270.qasm',
        'alu-bdd_288.qasm',
        'decod24-v1_41.qasm',
        '4gt5_76.qasm',
        '4gt13_91.qasm',
        '4gt13_90.qasm',
        'alu-v4_36.qasm',
        '4gt5_77.qasm',
        'one-two-three-v1_99.qasm',
        'rd53_138.qasm',
        'one-two-three-v0_98.qasm',
        '4gt10-v1_81.qasm',
        'decod24-v3_45.qasm',
        'aj-e11_165.qasm',
        '4mod7-v0_94.qasm',
        'alu-v2_32.qasm',
        '4mod7-v1_96.qasm',
        'cnt3-5_179.qasm',
        'mod10_176.qasm',
        '4gt4-v0_80.qasm',
        '4gt12-v0_88.qasm',
        '0410184_169.qasm',
        '4_49_16.qasm',
        '4gt12-v1_89.qasm',
        '4gt4-v0_79.qasm',
        'hwb4_49.qasm',
        '4gt4-v0_78.qasm',
        'mod10_171.qasm',
        '4gt12-v0_87.qasm',
        '4gt12-v0_86.qasm',
        '4gt4-v0_72.qasm',
        '4gt4-v1_74.qasm',
        'mini-alu_167.qasm',
        'one-two-three-v0_97.qasm',
        'rd53_135.qasm',
        'ham7_104.qasm',
        'decod24-enable_126.qasm',
        'mod8-10_178.qasm',
        '4gt4-v0_73.qasm',
        'ex3_229.qasm',
        'mod8-10_177.qasm',
        'alu-v2_31.qasm',
        'C17_204.qasm',
        'rd53_131.qasm',
        'alu-v2_30.qasm',
        'mod5adder_127.qasm',
        'rd53_133.qasm',
        'majority_239.qasm',
        'ex2_227.qasm',
        'cm82a_208.qasm',
        'sf_276.qasm',
        'sf_274.qasm',
        'con1_216.qasm',
        'rd53_130.qasm',
        'f2_232.qasm',
        'rd53_251.qasm',
        'hwb5_53.qasm',
        'radd_250.qasm',
        'rd73_252.qasm',
        'cycle10_2_110.qasm',
        'hwb6_56.qasm',
        'cm85a_209.qasm',
    ),
    "large": (
        #    'rd84_253.qasm',
        #     'root_255.qasm',
        #     'mlp4_245.qasm',
        #     'urf2_277.qasm',
        'sym9_148.qasm',
        # 'hwb7_59.qasm',
        # 'clip_206.qasm',
        # 'sym9_193.qasm',
        # 'dist_223.qasm',
        # 'sao2_257.qasm',
        'urf5_280.qasm',
        'urf1_278.qasm',
        'sym10_262.qasm',
        'hwb8_113.qasm',
    ),
}

initial_map_lists = {
    "mcts_small": [
        [19, 18, 13, 8, 4, 1, 10, 12, 14, 9, 3, 2, 6, 11, 17, 16, 5, 7, 15, 0],
        [17, 3, 0, 10, 11, 5, 2, 8, 13, 12, 1, 7, 6, 18, 16, 4, 9, 19, 15, 14],
        [5, 15, 13, 18, 19, 6, 10, 16, 8, 7, 12, 17, 11, 9, 3, 2, 0, 4, 14, 1],
        [11, 6, 4, 1, 2, 8, 3, 13, 9, 7, 12, 15, 18, 14, 16, 0, 5, 10, 17, 19],
        [7, 13, 12, 6, 1, 8, 11, 19, 5, 18, 16, 14, 2, 4, 15, 0, 17, 9, 3, 10],
        [2, 1, 10, 3, 8, 16, 5, 17, 11, 13, 7, 6, 12, 18, 14, 9, 0, 4, 19, 15],
        [1, 6, 9, 12, 3, 8, 2, 4, 13, 7, 16, 10, 19, 11, 14, 18, 0, 5, 15, 17],
        [13, 16, 17, 12, 1, 10, 5, 8, 11, 6, 7, 2, 9, 19, 3, 14, 18, 15, 4, 0],
        [6, 7, 13, 2, 1, 8, 18, 0, 11, 10, 16, 12, 17, 5, 19, 4, 9, 3, 14, 15],
        [17, 16, 13, 8, 1, 2, 6, 11, 7, 12, 15, 5, 18, 0, 10, 14, 3, 9, 4, 19],
        [8, 2, 0, 5, 13, 16, 10, 6, 1, 11, 12, 7, 17, 15, 19, 4, 14, 9, 3, 18]]
    ,
    "extended": [
        [14, 19, 13, 8, 3, 2, 17, 1, 15, 9, 18, 16, 5, 11, 10, 4, 6, 12, 7, 0],
        [7, 13, 1, 12, 6, 2, 5, 19, 4, 15, 17, 9, 18, 11, 10, 14, 3, 0, 8, 16],
        [12, 13, 8, 7, 11, 17, 15, 6, 10, 18, 2, 16, 4, 3, 19, 5, 1, 0, 9, 14],
        [7, 12, 8, 6, 13, 11, 2, 17, 3, 16, 1, 18, 5, 10, 9, 14, 0, 4, 15, 19],
        [8, 12, 7, 10, 17, 6, 14, 0, 9, 15, 2, 16, 11, 18, 13, 3, 5, 4, 19, 1],
        [8, 12, 7, 13, 6, 11, 2, 17, 1, 9, 14, 16, 3, 18, 5, 10, 0, 4, 15, 19],
        [13, 16, 12, 11, 17, 5, 7, 14, 10, 0, 3, 1, 2, 9, 8, 4, 15, 6, 18, 19],
        [13, 6, 12, 7, 8, 11, 2, 17, 1, 16, 3, 15, 5, 10, 9, 14, 0, 4, 18, 19],
        [7, 8, 2, 4, 12, 17, 1, 15, 6, 0, 16, 9, 10, 11, 19, 14, 3, 5, 13, 18],
        [18, 14, 9, 8, 13, 4, 10, 0, 3, 16, 11, 12, 5, 7, 1, 2, 19, 15, 17, 6],
        [12, 8, 3, 2, 7, 17, 0, 16, 4, 1, 19, 9, 15, 13, 18, 10, 6, 11, 14, 5],
        [8, 7, 13, 12, 6, 10, 2, 17, 1, 16, 3, 18, 5, 11, 9, 14, 0, 4, 15, 19],
        [8, 6, 13, 7, 12, 11, 2, 10, 1, 16, 3, 18, 5, 17, 9, 14, 0, 4, 15, 19],
        [6, 12, 1, 7, 2, 19, 8, 9, 16, 15, 4, 17, 5, 3, 10, 18, 0, 13, 11, 14],
        [6, 10, 5, 11, 13, 4, 8, 12, 0, 7, 1, 3, 19, 9, 16, 18, 2, 15, 17, 14],
        [8, 2, 7, 6, 13, 17, 14, 11, 19, 4, 5, 1, 15, 16, 0, 12, 18, 3, 10, 9],
        [8, 7, 12, 13, 2, 11, 6, 17, 1, 16, 3, 18, 5, 10, 9, 14, 0, 4, 15, 19],
        [13, 12, 7, 16, 8, 6, 5, 10, 14, 1, 4, 3, 2, 17, 9, 0, 11, 15, 18, 19],
        [2, 6, 10, 5, 1, 3, 17, 13, 11, 14, 8, 19, 9, 0, 12, 7, 18, 15, 4, 16],
        [6, 11, 12, 5, 16, 2, 9, 8, 7, 18, 15, 13, 17, 10, 1, 14, 0, 4, 3, 19],
        [1, 11, 6, 7, 10, 8, 3, 4, 15, 19, 5, 9, 13, 0, 18, 12, 16, 14, 17, 2],
        [1, 5, 6, 11, 7, 16, 3, 18, 8, 10, 12, 4, 9, 14, 2, 0, 13, 17, 15, 19],
        [1, 12, 7, 8, 2, 15, 10, 17, 14, 11, 4, 19, 5, 3, 16, 0, 6, 18, 9, 13],
        [8, 4, 9, 14, 0, 12, 13, 6, 11, 2, 17, 1, 16, 3, 18, 15, 10, 7, 5, 19],
        [7, 2, 6, 1, 11, 12, 15, 3, 8, 10, 19, 4, 0, 9, 18, 13, 16, 14, 17, 5],
        [2, 10, 6, 7, 11, 1, 13, 19, 17, 8, 14, 15, 18, 9, 12, 4, 16, 5, 3, 0],
        [16, 17, 11, 12, 10, 13, 2, 15, 0, 14, 19, 3, 8, 1, 18, 4, 7, 9, 5, 6],
        [10, 2, 6, 5, 11, 16, 1, 9, 14, 0, 18, 13, 17, 15, 3, 4, 19, 7, 12, 8],
        [8, 7, 11, 17, 12, 9, 4, 6, 18, 2, 14, 13, 1, 3, 19, 16, 5, 15, 0, 10],
        [13, 19, 12, 8, 14, 9, 5, 18, 16, 11, 17, 6, 1, 10, 0, 15, 4, 7, 2, 3],
        [13, 8, 7, 12, 3, 0, 18, 9, 4, 11, 16, 15, 19, 2, 5, 14, 17, 10, 6, 1],
        [12, 7, 13, 3, 8, 11, 2, 17, 1, 16, 15, 18, 5, 10, 9, 14, 0, 4, 6, 19],
        [13, 12, 8, 6, 7, 11, 10, 9, 14, 2, 17, 1, 16, 3, 18, 5, 0, 4, 15, 19],
        [13, 6, 12, 7, 8, 11, 2, 17, 1, 16, 3, 18, 5, 10, 14, 9, 0, 4, 15, 19],
        [13, 3, 2, 4, 8, 6, 7, 15, 16, 5, 14, 0, 19, 17, 18, 10, 12, 11, 1, 9],
        [11, 6, 17, 5, 7, 10, 19, 3, 8, 13, 15, 16, 14, 2, 9, 12, 18, 4, 0, 1],
        [6, 7, 12, 8, 13, 11, 17, 1, 16, 3, 18, 5, 10, 2, 9, 14, 0, 4, 15, 19],
        [12, 8, 11, 16, 13, 3, 15, 17, 7, 0, 18, 5, 14, 19, 10, 1, 6, 9, 2, 4],
        [11, 10, 5, 6, 0, 4, 18, 3, 16, 19, 2, 13, 8, 12, 14, 7, 17, 15, 1, 9],
        [7, 12, 18, 13, 9, 8, 14, 10, 15, 11, 5, 3, 4, 0, 1, 17, 19, 6, 2, 16],
        [14, 13, 8, 7, 12, 4, 11, 15, 18, 17, 1, 3, 9, 16, 19, 10, 0, 5, 6, 2],
        [13, 7, 2, 6, 1, 19, 4, 14, 0, 15, 16, 8, 11, 3, 5, 9, 18, 12, 10, 17],
        [16, 13, 8, 7, 12, 6, 15, 9, 17, 4, 11, 19, 5, 1, 10, 14, 0, 3, 2, 18],
        [2, 5, 11, 10, 6, 15, 12, 8, 7, 17, 9, 3, 13, 19, 4, 18, 1, 0, 16, 14],
        [2, 13, 7, 6, 1, 5, 17, 14, 8, 16, 19, 9, 3, 12, 11, 15, 10, 18, 4, 0],
        [3, 8, 4, 12, 13, 9, 1, 2, 7, 0, 5, 17, 6, 11, 14, 15, 16, 19, 18, 10],
        [12, 7, 8, 16, 17, 2, 1, 9, 14, 3, 19, 15, 5, 0, 13, 18, 11, 4, 6, 10],
        [6, 10, 5, 16, 8, 11, 12, 13, 15, 19, 7, 2, 1, 0, 14, 9, 3, 17, 18, 4],
        [17, 16, 12, 13, 11, 5, 0, 14, 1, 8, 7, 4, 6, 19, 9, 15, 10, 3, 2, 18],
        [13, 19, 14, 18, 7, 15, 16, 2, 9, 11, 1, 12, 4, 5, 10, 17, 3, 0, 6, 8],
        [12, 8, 2, 1, 7, 17, 0, 6, 13, 4, 9, 3, 19, 15, 5, 11, 14, 16, 18, 10],
        [10, 0, 5, 6, 11, 15, 7, 12, 3, 19, 4, 13, 16, 18, 17, 1, 8, 2, 14, 9],
        [13, 12, 16, 11, 7, 3, 8, 15, 0, 17, 2, 19, 6, 1, 18, 9, 14, 10, 5, 4],
        [7, 8, 12, 19, 13, 9, 3, 6, 14, 4, 17, 15, 16, 18, 10, 5, 0, 1, 11, 2],
        [9, 8, 7, 4, 3, 10, 12, 1, 2, 18, 14, 5, 0, 17, 19, 6, 11, 16, 13, 15],
        [9, 4, 3, 13, 8, 19, 12, 16, 17, 11, 10, 5, 6, 1, 7, 2, 14, 15, 0, 18],
        [7, 4, 13, 8, 12, 15, 10, 16, 5, 6, 18, 17, 11, 2, 9, 19, 3, 14, 0, 1],
        [8, 12, 7, 13, 14, 19, 18, 2, 4, 3, 11, 1, 17, 9, 10, 6, 15, 16, 0, 5],
        [7, 12, 8, 13, 19, 14, 1, 9, 18, 10, 3, 5, 6, 4, 17, 15, 16, 2, 11, 0],
        [4, 3, 8, 13, 7, 2, 6, 10, 11, 16, 12, 19, 14, 18, 15, 0, 1, 9, 17, 5],
        [11, 7, 5, 10, 6, 1, 16, 18, 9, 12, 13, 17, 19, 15, 3, 14, 2, 4, 0, 8],
        [5, 11, 1, 7, 2, 6, 4, 10, 3, 13, 16, 15, 9, 19, 18, 0, 17, 8, 12, 14],
        [13, 12, 8, 3, 9, 7, 19, 5, 0, 16, 4, 1, 18, 6, 14, 2, 15, 11, 10, 17],
        [19, 7, 8, 13, 12, 0, 16, 2, 6, 4, 3, 14, 10, 9, 11, 17, 5, 18, 15, 1],
        [7, 2, 6, 11, 5, 1, 10, 9, 0, 17, 12, 15, 16, 19, 13, 18, 3, 8, 14, 4],
        [9, 3, 4, 12, 8, 5, 16, 2, 15, 18, 6, 19, 11, 7, 14, 13, 10, 1, 0, 17],
        [12, 8, 13, 14, 18, 7, 15, 6, 0, 3, 11, 16, 1, 4, 10, 9, 19, 17, 5, 2],
        [1, 6, 7, 12, 8, 2, 9, 10, 15, 3, 14, 18, 4, 0, 13, 5, 11, 16, 17, 19],
        [7, 8, 14, 18, 12, 13, 11, 3, 10, 1, 19, 16, 5, 2, 6, 9, 0, 4, 15, 17],
        [12, 11, 2, 6, 7, 1, 0, 4, 3, 10, 19, 18, 15, 8, 9, 14, 17, 16, 5, 13],
        [16, 11, 8, 17, 12, 0, 14, 9, 10, 19, 18, 7, 3, 6, 13, 2, 4, 15, 5, 1],
        [13, 8, 12, 11, 7, 10, 0, 16, 1, 2, 15, 9, 6, 18, 17, 3, 19, 5, 14, 4],
        [13, 7, 8, 18, 14, 12, 4, 5, 19, 11, 1, 0, 16, 9, 2, 3, 15, 6, 10, 17],
        [16, 0, 18, 14, 7, 13, 19, 1, 3, 4, 2, 6, 15, 10, 5, 9, 8, 11, 12, 17],
        [5, 6, 7, 11, 12, 16, 17, 10, 18, 2, 0, 9, 19, 13, 3, 1, 15, 8, 14, 4],
        [6, 17, 10, 12, 11, 16, 8, 0, 18, 7, 2, 5, 3, 9, 19, 15, 1, 4, 14, 13],
        [19, 13, 8, 7, 18, 14, 5, 16, 3, 6, 4, 1, 17, 2, 9, 11, 12, 0, 15, 10],
        [8, 13, 17, 11, 7, 12, 16, 6, 3, 14, 15, 5, 9, 10, 2, 18, 19, 0, 4, 1],
        [5, 10, 1, 6, 7, 2, 0, 19, 12, 3, 15, 8, 16, 4, 9, 14, 13, 11, 17, 18],
        [8, 4, 3, 9, 12, 11, 10, 14, 17, 15, 6, 13, 5, 18, 2, 0, 1, 16, 7, 19],
        [1, 10, 0, 5, 7, 2, 6, 8, 12, 16, 18, 9, 15, 14, 4, 11, 13, 3, 17, 19],
        [18, 7, 8, 12, 17, 13, 14, 1, 9, 16, 15, 6, 0, 2, 5, 10, 4, 19, 11, 3],
        [13, 7, 3, 4, 9, 8, 0, 16, 5, 11, 10, 14, 6, 2, 18, 17, 15, 19, 12, 1],
        [12, 8, 7, 2, 6, 1, 18, 3, 11, 15, 17, 0, 14, 19, 4, 9, 10, 16, 13, 5],
        [12, 16, 6, 11, 10, 5, 0, 13, 3, 14, 15, 9, 18, 2, 7, 4, 17, 19, 1, 8],
        [4, 9, 7, 12, 13, 11, 8, 10, 19, 5, 2, 16, 3, 15, 1, 6, 18, 14, 17, 0],
        [19, 8, 17, 11, 12, 13, 18, 4, 2, 5, 3, 7, 0, 14, 10, 9, 6, 15, 16, 1],
        [18, 16, 12, 13, 8, 7, 11, 17, 9, 3, 15, 14, 5, 1, 19, 6, 10, 0, 2, 4],
        [1, 7, 13, 8, 6, 2, 10, 16, 5, 17, 4, 9, 12, 11, 3, 19, 18, 14, 15, 0],
        [13, 12, 11, 17, 8, 7, 5, 3, 9, 2, 4, 15, 0, 1, 18, 16, 14, 6, 10, 19],
        [13, 5, 8, 6, 11, 12, 7, 16, 17, 4, 15, 9, 3, 14, 18, 2, 19, 0, 10, 1],
        [6, 5, 12, 11, 10, 17, 16, 14, 0, 15, 8, 9, 18, 1, 2, 13, 3, 7, 4, 19],
        [4, 13, 7, 8, 9, 12, 18, 14, 6, 3, 15, 16, 10, 2, 0, 19, 5, 1, 11, 17],
        [6, 9, 1, 13, 2, 8, 12, 7, 16, 5, 4, 0, 18, 17, 15, 19, 3, 11, 10, 14],
        [4, 9, 13, 7, 12, 8, 18, 17, 11, 19, 10, 15, 16, 5, 0, 14, 2, 1, 3, 6],
        [10, 0, 13, 3, 15, 5, 1, 8, 12, 11, 7, 2, 6, 18, 4, 19, 16, 14, 17, 9],
        [10, 5, 16, 6, 8, 13, 7, 17, 12, 11, 3, 15, 1, 0, 2, 18, 14, 19, 4, 9],
        [18, 2, 6, 8, 16, 10, 5, 17, 11, 12, 7, 13, 0, 4, 3, 15, 1, 14, 19, 9],
        [17, 12, 6, 10, 5, 16, 11, 19, 2, 1, 14, 4, 9, 18, 3, 8, 7, 13, 0, 15],
        [3, 1, 15, 2, 6, 8, 13, 16, 17, 11, 10, 12, 5, 7, 4, 9, 0, 19, 18, 14],
    ],
    "large": [
        # [10, 1, 3, 8, 5, 16, 13, 7, 2, 12, 11, 6, 0, 17, 14, 19, 15, 18, 4, 9],
        # [1, 2, 8, 13, 18, 5, 17, 16, 12, 7, 11, 10, 6, 4, 3, 19, 9, 0, 15, 14],
        # [9, 15, 3, 18, 13, 2, 5, 1, 8, 10, 6, 7, 11, 12, 16, 17, 4, 0, 14, 19],
        # [13, 11, 6, 1, 8, 12, 2, 7, 19, 3, 15, 17, 5, 10, 9, 0, 16, 18, 4, 14],
        [19, 18, 14, 17, 8, 13, 12, 7, 6, 2, 5, 9, 0, 1, 15, 4, 11, 10, 3, 16],
        # [11, 6, 1, 12, 8, 13, 2, 7, 19, 0, 18, 5, 15, 3, 10, 16, 4, 9, 14, 17],
        # [10, 3, 0, 5, 16, 6, 17, 8, 13, 12, 11, 7, 2, 1, 4, 18, 14, 9, 15, 19],
        # [17, 16, 13, 10, 1, 2, 5, 6, 7, 12, 11, 15, 19, 18, 8, 3, 14, 4, 9, 0],
        # [2, 3, 10, 5, 17, 13, 8, 16, 11, 6, 12, 7, 1, 18, 15, 4, 19, 9, 0, 14],
        # [10, 5, 2, 15, 1, 8, 13, 18, 16, 11, 17, 12, 7, 6, 19, 9, 4, 0, 3, 14],
        [19, 17, 16, 8, 14, 18, 7, 12, 13, 5, 15, 2, 11, 3, 0, 9, 6, 1, 10, 4],
        [8, 12, 11, 1, 13, 3, 2, 6, 7, 19, 10, 0, 18, 15, 9, 16, 5, 4, 17, 14],
        [17, 5, 16, 13, 8, 1, 10, 2, 7, 12, 11, 6, 15, 3, 9, 14, 0, 19, 18, 4],
    ]
}
