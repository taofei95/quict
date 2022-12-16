from collections import deque

import numpy as np

from QuICT.core.gate import BasicGate
from QuICT.core.utils import GateType

from QuICT.core import Circuit
from QuICT.core.utils.circuit_info import CircuitMode


DEBUG = False


class CircuitPartitionOptimization(object):
    def __init__(self,
                 optimize_methods=None,
                 partition_method='circuit_based'):

        assert partition_method in ['circuit_based', 'dag_based'], \
            "partition_method can only be 'circuit_based', 'dag_based'"

        if optimize_methods is None:
            optimize_methods = {}

        self.partition_method = partition_method
        self.optimize_methods = optimize_methods

    class SubCircuit:
        def __init__(self, width, mode):
            self.circuit = Circuit(width)
            self.mode = mode
            # self.occupied_qubits = set()
            # self.blocked_qubits = set()
            # self.last_gate_label = -1
            self.succ = set()

        def append(self, g: BasicGate, label: int):
            args = set(g.cargs + g.targs)
            # if self.blocked_qubits & args:
            #     return False
            self.circuit.append(g)
            # self.last_gate_label = max(self.last_gate_label, label)
            # return True

        def block(self, qubits, succ_block):
            self.succ.add(succ_block)
            # self.blocked_qubits |= set(qubits)

        def size(self):
            return self.circuit.size()

    def _get_init_mode(self, g: BasicGate):
        if g.type == GateType.ccx:
            return CircuitMode.Arithmetic
        if g.type in [GateType.rz, GateType.t, GateType.tdg]:
            return CircuitMode.CliffordRz
        if g.is_clifford():
            return CircuitMode.Clifford
        return CircuitMode.Misc

    def _get_result_mode(self, mode_1, g: BasicGate):
        if mode_1 == CircuitMode.CliffordRz and g.is_clifford():
            return CircuitMode.CliffordRz
        if mode_1 == CircuitMode.Arithmetic and g.type in [GateType.x, GateType.cx]:
            return CircuitMode.Arithmetic
        if mode_1 == self._get_init_mode(g):
            return mode_1
        return None

    def _get_merged_mode(self, prev: SubCircuit, succ: SubCircuit, threshold: int):
        if prev.mode == succ.mode:
            return prev.mode
        if min(prev.circuit.size(), succ.circuit.size()) <= threshold:
            if {prev.mode, succ.mode} == {CircuitMode.Clifford, CircuitMode.CliffordRz}:
                return CircuitMode.CliffordRz
            elif prev.size() >= succ.size() and prev.mode == CircuitMode.CliffordRz:
                return CircuitMode.CliffordRz
            elif succ.size() >= prev.size() and succ.mode == CircuitMode.CliffordRz:
                return CircuitMode.CliffordRz
        return None

    def _topo_sort(self, n, topo_edges):
        out = [[] for i in range(n)]
        deg = [0] * n
        res = []
        for u, v in topo_edges:
            out[u].append(v)
            deg[v] += 1

        que = deque(filter(lambda x: deg[x] == 0, range(n)))
        while que:
            cur = que.popleft()
            res.append(cur)
            for nxt in out[cur]:
                deg[nxt] -= 1
                if deg[nxt] == 0:
                    que.append(nxt)

        # print(topo_edges)
        # print(res)
        return res

    def _topo_sort_blocks(self, blocks):
        deg = [0] * len(blocks)
        for idx, sub in enumerate(blocks):
            for jdx in sub.succ:
                deg[jdx] += 1

        res = []
        que = deque(filter(lambda x: deg[x] == 0, range(len(blocks))))
        while que:
            cur = que.popleft()
            res.append(cur)
            for nxt in blocks[cur].succ:
                deg[nxt] -= 1
                if deg[nxt] == 0:
                    que.append(nxt)

        return [blocks[i] for i in res]

    def _is_reachable(self, src, dest, blocks):
        que = deque([src])
        cnt = 0
        while que:
            cur = que.popleft()
            if cnt >= 100:
                pass
            cnt += 1
            if cur == dest:
                return True
            for nxt in blocks[cur].succ:
                que.append(nxt)
        return False

    def _circuit_based_partition(self, circuit: Circuit):
        cur_circ = np.array([-1] * circuit.size(), dtype=np.int)
        start = {}
        blocks = []

        topo_edges = []

        # partition into strict small blocks
        for idx, g in enumerate(circuit.gates):
            g: BasicGate
            # find arithmetic candidate
            targ_mode = self._get_init_mode(g)
            args = g.cargs + g.targs
            prev = cur_circ[args]
            if np.all(prev == -1):
                if targ_mode not in start:
                    blocks.append(self.SubCircuit(circuit.width(), targ_mode))
                    start[targ_mode] = len(blocks) - 1

                id_ = start[targ_mode]
                blocks[id_].append(g.copy(), idx)
                cur_circ[args] = id_
                continue

            cur_id = -1
            for i in filter(lambda n: cur_circ[n] != -1, args):
                sub_circ = blocks[cur_circ[i]]
                result_mode = self._get_result_mode(sub_circ.mode, g)
                if result_mode is None:
                    continue

                flag = True
                for j in filter(lambda n: cur_circ[n] != -1, args):
                    if cur_circ[j] != cur_circ[i] and \
                            self._is_reachable(cur_circ[i], cur_circ[j], blocks):
                        flag = False
                        break

                if flag:
                    sub_circ.append(g, idx)
                    cur_id = cur_circ[i]
                    sub_circ.mode = result_mode
                    break

            if cur_id == -1:
                blocks.append(self.SubCircuit(circuit.width(), targ_mode))
                cur_id = len(blocks) - 1
                blocks[cur_id].append(g, idx)

            for j in args:
                if cur_circ[j] != -1 and cur_circ[j] != cur_id:
                    blocks[cur_circ[j]].succ.add(cur_id)
                    # blocks[cur_circ[j]].block(args, cur_id)
                    topo_edges.append((cur_circ[j], cur_id))

            cur_circ[args] = cur_id

        # args = self._topo_sort(len(blocks), topo_edges)
        # blocks = [blocks[i] for i in args]
        blocks = self._topo_sort_blocks(blocks)

        if DEBUG:
            circuit.draw(filename='circ_before.jpg')
            v_circ = Circuit(circuit.width())
            for idx, sub in enumerate(blocks, 1):
                # sub.circuit.draw(filename=f'd{idx}_{sub.mode}.jpg')
                v_circ.extend(sub.circuit.gates)
            v_circ.draw(filename='circ_small_blocks.jpg')
            assert np.allclose(v_circ.matrix(), circuit.matrix()), 'small blocks incorrect'

        # merge blocks
        threshold = 10
        for i in range(1, len(blocks)):
            merged_mode = self._get_merged_mode(blocks[i - 1], blocks[i], threshold)
            if merged_mode is not None:
                blocks[i - 1].circuit.extend(blocks[i].circuit.gates)
                blocks[i - 1].mode = merged_mode
                blocks[i] = blocks[i - 1]
                blocks[i - 1] = None

        return list(filter(lambda x: x is not None, blocks))

    def _dag_based_partition(self, circuit):
        assert False, 'not supported yet'

    def partition(self, circuit):
        if self.partition_method == 'circuit_based':
            return self._circuit_based_partition(circuit)
        elif self.partition_method == 'circuit_based':
            return self._dag_based_partition(circuit)

        return [circuit]

    def execute(self, circuit, verbose=True):
        circ_list = self.partition(circuit)

        if DEBUG:
            v_circ = Circuit(circuit.width())
            for idx, each in enumerate(circ_list, 1):
                v_circ.extend(each.circuit.gates)
                each.circuit.draw(filename=f'c{idx}_{each.mode}.jpg')
            v_circ.draw(filename='circ_big_blocks.jpg')
            assert np.allclose(v_circ.matrix(), circuit.matrix()), 'large blocks incorrect'

        new_circ = Circuit(circuit.width())
        for idx, each in enumerate(circ_list, 1):
            if verbose:
                print(f'sub-circuit #{idx}: {each.mode}, {each.circuit.size()}')
            cur_circ = each.circuit
            if each.mode in self.optimize_methods:
                for optimizer in self.optimize_methods[each.mode]:
                    cur_circ = optimizer.execute(cur_circ)
                    if verbose:
                        print(f'\tafter {optimizer}: {cur_circ.size()}')
            new_circ.extend(cur_circ.gates)

        return new_circ

    def add_optimizer(self, circuit_mode: CircuitMode, optimizer):
        if circuit_mode not in self.optimize_methods:
            self.optimize_methods[circuit_mode] = []
        self.optimize_methods[circuit_mode].append(optimizer)
