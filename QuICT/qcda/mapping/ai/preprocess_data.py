from typing import Dict, List, Set, Tuple

from QuICT.core import *
from QuICT.core.gate.gate import BasicGate
from QuICT.core.utils.gate_type import GateType
from QuICT.tools.interface.qasm_interface import OPENQASMInterface


class Preprocessor:
    def __init__(self) -> None:
        pass

    @classmethod
    def qasm_to_circ(cls, qasm_file_path: str) -> Circuit:
        instance = OPENQASMInterface.load_file(qasm_file_path)
        return instance.circuit

    @classmethod
    def circ_to_dag(cls, circ: Circuit) -> List[Tuple[int, int]]:
        """Convert a Circuit into DAG. Only 2-bit gates are included.
        Gate types and parameters are stripped.
        """
        n_qubit = circ.width()
        node_cnt = -1
        edges = []
        occupy = [-1 for _ in range(n_qubit)]
        for gate in circ.gates:
            gate: BasicGate
            if gate.is_single():
                continue
            if gate.controls + gate.targets > 2:
                raise NotImplementedError("Gate with qubit >= 3 is not supported.")
            node_cnt += 1
            args = gate.cargs + gate.targs
            a, b = args[0], args[1]
            if occupy[a] == -1 and occupy[b] == -1:
                pass
            elif occupy[a] == occupy[b]:
                edges.append((occupy[a], node_cnt))
            elif occupy[a] == -1:
                edges.append((occupy[b], node_cnt))
            elif occupy[b] == -1:
                edges.append((occupy[a], node_cnt))
            else:
                edges.append((occupy[a], node_cnt))
                edges.append((occupy[b], node_cnt))
            occupy[a] = node_cnt
            occupy[b] = node_cnt
        return edges

    @classmethod
    def circ_to_layers(cls, circ: Circuit) -> List[List[BasicGate]]:
        n_qubit = circ.width()
        layers = []
        layer_cnt = 0
        occupy = [-1 for _ in range(n_qubit)]
        for gate in circ.gates:
            gate: BasicGate
            occ_layer = -1
            for arg in gate.cargs + gate.targs:
                occ_layer = max(occ_layer, occupy[arg])
            if occ_layer == layer_cnt:
                layer_cnt += 1
                layers.append([])
            layers[occ_layer].append(gate)
        return layers

    @classmethod
    def circ_to_swap_dist(cls, circ: Circuit) -> Dict[Set[int], float]:
        ans: Dict[Set[int], float] = dict()
        layers = cls.circ_to_layers(circ)
        for layer in layers:
            has_swap = False
            for gate in layer:
                if gate.type == GateType.swap:
                    has_swap = True
                    break
            if has_swap:
                n_swap = 0
                for gate in layer:
                    if gate.type == GateType.swap:
                        n_swap += 1
                for gate in layer:
                    if gate.type == GateType.swap:
                        ans[set(gate.targs)] = 1.0 / n_swap
        return ans
