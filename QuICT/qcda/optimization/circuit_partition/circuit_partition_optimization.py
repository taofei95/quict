from collections import deque

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import BasicGate
from QuICT.core.utils import GateType
from QuICT.core.utils.circuit_info import CircuitMode


class CircuitPartitionOptimization(object):
    """
    A meta optimizer that partitions circuits into different blocks and applies
    corresponding optimizers.
    """
    def __init__(self, optimize_methods=None, merge_threshold=10, verbose=False):
        """
        Args:
            optimize_methods(dict): optimize_methods[m] is a list of methods that
                are applied to a block of mode m.
            merge_threshold(int): a heuristic parameter controlling block size.
        """

        if optimize_methods is None:
            optimize_methods = {}

        self.partition_method = 'circuit_based'
        self.optimize_methods = optimize_methods
        self.merge_threshold = merge_threshold
        self.verbose = verbose

    def __repr__(self):
        return f'CircuitPartitionOptimization(partition_method={self.partition_method}, ' \
               f'optimize_methods={self.optimize_methods}, ' \
               f'merge_threshold={self.merge_threshold})'

    class SubCircuit:
        """
        A class for circuit blocks.
        """
        def __init__(self, width, mode):
            self.circuit = Circuit(width)
            self.mode = mode
            self.succ = set()

        def append(self, g: BasicGate):
            self.circuit.append(g)

        def extend(self, gates):
            self.circuit.extend(gates)

        def size(self):
            return self.circuit.size()

    def _get_init_mode(self, g: BasicGate):
        """
        Returns:
            CircuitMode: the mode of a block starting with gate `g`
        """
        if g.type == GateType.ccx:
            return CircuitMode.Arithmetic
        if g.type in [GateType.rz, GateType.t, GateType.tdg]:
            return CircuitMode.CliffordRz
        if g.is_clifford():
            return CircuitMode.Clifford
        return CircuitMode.Misc

    def _get_appended_mode(self, mode_prev, g: BasicGate):
        """
        Returns:
            CircuitMode: the mode if gate `g` is appended to a block of mode `mode_prev`
        """
        if mode_prev == self._get_init_mode(g):
            return mode_prev
        if mode_prev == CircuitMode.CliffordRz and g.is_clifford():
            return CircuitMode.CliffordRz
        if mode_prev == CircuitMode.Arithmetic and g.type in [GateType.x, GateType.cx, GateType.ccx]:
            return CircuitMode.Arithmetic
        return None

    def _get_merged_mode(self, prev: SubCircuit, succ: SubCircuit):
        """
        Returns:
            CircuitMode: the mode if `prev` and `succ` is merged
        """
        if prev.mode == succ.mode:
            return prev.mode

        robust_mode = {CircuitMode.CliffordRz, CircuitMode.Misc}
        if min(prev.circuit.size(), succ.circuit.size()) <= self.merge_threshold:
            if {prev.mode, succ.mode} == {CircuitMode.Clifford, CircuitMode.CliffordRz}:
                return CircuitMode.CliffordRz
            elif prev.size() >= succ.size() and prev.mode in robust_mode:
                return prev.mode
            elif succ.size() >= prev.size() and succ.mode in robust_mode:
                return succ.mode
        return None

    def _topo_sort_blocks(self, blocks):
        """
        Sort `blocks` in topological order.
        """

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
        """
        Returns:
            bool: if block `dest` is reachable from block `src`
        """

        que = deque([src])
        cnt = 0
        while que:
            cur = que.popleft()
            cnt += 1
            if cur == dest:
                return True
            for nxt in blocks[cur].succ:
                que.append(nxt)
        return False

    def _circuit_based_partition(self, circuit: Circuit):
        """
        Partition circuit into blocks based on circuit net-list.

        Returns:
            List[SubCircuit]: a list of blocks
        """

        # cur_circ[i]: current SubCircuit id on qubit i
        # start[m]: the left-most block of type m
        # blocks[i]: SubCircuit i
        cur_circ = np.array([-1] * circuit.size(), dtype=np.int)
        start = {}
        blocks = []

        # partition into strict small blocks
        for idx, g in enumerate(circuit.gates):
            init_mode = self._get_init_mode(g)
            args = g.cargs + g.targs
            prev = cur_circ[args]

            # if no gates proceeds g, add into a left-most block
            if np.all(prev == -1):
                if init_mode not in start:
                    blocks.append(self.SubCircuit(circuit.width(), init_mode))
                    start[init_mode] = len(blocks) - 1

                id_ = start[init_mode]
                blocks[id_].append(g.copy())
                cur_circ[args] = id_
                continue

            # find a suitable block to append
            cur_id = -1
            for i in filter(lambda n: cur_circ[n] != -1, args):
                sub_circ = blocks[cur_circ[i]]

                # condition 1: gate type is compatible
                appended_mode = self._get_appended_mode(sub_circ.mode, g)
                if appended_mode is None:
                    continue

                # condition 2: appending to block i forms no cycle
                flag = True
                for j in filter(lambda n: cur_circ[n] != -1, args):
                    if cur_circ[j] != cur_circ[i] and \
                            self._is_reachable(cur_circ[i], cur_circ[j], blocks):
                        flag = False
                        break

                if flag:
                    sub_circ.append(g)
                    cur_id = cur_circ[i]
                    sub_circ.mode = appended_mode
                    break

            # no existing block is suitable, create a new block
            if cur_id == -1:
                blocks.append(self.SubCircuit(circuit.width(), init_mode))
                cur_id = len(blocks) - 1
                blocks[cur_id].append(g)

            for j in args:
                if cur_circ[j] != -1 and cur_circ[j] != cur_id:
                    blocks[cur_circ[j]].succ.add(cur_id)

            cur_circ[args] = cur_id

        # topological sort blocks
        blocks = self._topo_sort_blocks(blocks)

        # merge blocks
        for i in range(1, len(blocks)):
            merged_mode = self._get_merged_mode(blocks[i - 1], blocks[i])
            if merged_mode is not None:
                blocks[i - 1].extend(blocks[i].circuit.gates)
                blocks[i - 1].mode = merged_mode
                blocks[i] = blocks[i - 1]
                blocks[i - 1] = None

        return list(filter(lambda x: x is not None, blocks))

    def _dag_based_partition(self, circuit):
        """
        Partition circuit into blocks based on DAG.
        """
        assert False, 'not supported yet'

    def partition(self, circuit):
        """
        Partition circuit into blocks.

        Returns:
            List[SubCircuit]: a list of blocks
        """

        if self.partition_method == 'circuit_based':
            return self._circuit_based_partition(circuit)
        elif self.partition_method == 'circuit_based':
            return self._dag_based_partition(circuit)

        return []

    def execute(self, circuit):
        """
        It will first partition the circuit into blocks and then apply optimizers to each block.

        Args:
            circuit(Circuit): the circuit to optimize
            verbose(bool): if verbose

        Returns:
            Circuit: the optimized circuit
        """
        circ_list = self.partition(circuit)

        new_circ = Circuit(circuit.width())
        for idx, each in enumerate(circ_list, 1):
            if self.verbose:
                print(f'sub-circuit #{idx}: {each.mode}, {each.circuit.size()}')
            cur_circ = each.circuit
            if each.mode in self.optimize_methods:
                for optimizer in self.optimize_methods[each.mode]:
                    cur_circ = optimizer.execute(cur_circ)
                    if self.verbose:
                        print(f'\tafter {optimizer}: {cur_circ.size()}')
            new_circ.extend(cur_circ.gates)

        return new_circ

    def add_optimizer(self, circuit_mode: CircuitMode, optimizer):
        """
        Append `optimizer` to the method list for mode `circuit_mode`

        Args:
            circuit_mode(CircuitMode): the mode
            optimizer: the optimizer instance
        """
        if circuit_mode not in self.optimize_methods:
            self.optimize_methods[circuit_mode] = []
        self.optimize_methods[circuit_mode].append(optimizer)
