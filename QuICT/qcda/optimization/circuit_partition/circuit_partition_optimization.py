from collections import deque
import random

import numpy as np

from QuICT.core import Circuit
from QuICT.core.circuit import DAGCircuit
from QuICT.core.gate import BasicGate
from QuICT.core.utils import GateType
from QuICT.core.utils.circuit_info import CircuitMode
from QuICT.qcda.optimization import SymbolicCliffordOptimization, CliffordRzOptimization, TemplateOptimization, \
    CommutativeOptimization
from QuICT.qcda.utility import OutputAligner


class CircuitPartitionOptimization(object):
    """
    A meta optimizer that partitions circuits into different blocks and applies
    corresponding optimizers.
    """

    def __init__(self,
                 level='light',
                 verbose=False,
                 keep_phase=False,
                 partition_method='dag_based',
                 optimize_methods=None):

        """
        Args:
            level(str): optimizing level (heavy or light). By default is light.
            keep_phase(bool): whether to keep the global phase as a GPhase gate in the output.
            partition_method(str): choose partition method (circuit_based or dag_based).
            optimize_methods(dict): Used if you want to customize sub optimizers.
                optimize_methods[m] is a list of methods that are applied to a block of mode m.
        """

        self.optimize_methods = optimize_methods
        if optimize_methods is None:
            self._add_default_optimizers(level, keep_phase)

        self.verbose = verbose
        self.keep_phase = keep_phase
        self.partition_method = partition_method

        # heuristic parameters
        self.merge_threshold = 30
        self.clifford_threshold = 0.6
        self.merge_iterations = 4

    def _add_default_optimizers(self, level, kp_ph):
        """
        Add default optimizers to self.optimize_methods.

        Args:
            level(str): optimizing level (heavy or light).
            kp_ph(bool): whether to keep the global phase as a GPhase gate in the output.
        """

        self.optimize_methods = {}
        if level == 'light':
            self.add_optimizer(CircuitMode.Clifford, SymbolicCliffordOptimization())
            self.add_optimizer(
                CircuitMode.CliffordRz,
                CliffordRzOptimization(level='light', keep_phase=kp_ph, optimize_toffoli=False)
            )
            self.add_optimizer(
                CircuitMode.Arithmetic,
                TemplateOptimization(template_typelist=[GateType.x, GateType.cx, GateType.ccx])
            )
            self.add_optimizer(CircuitMode.Misc, CommutativeOptimization(keep_phase=kp_ph))

        elif level == 'heavy':
            self.add_optimizer(CircuitMode.Clifford, SymbolicCliffordOptimization())
            self.add_optimizer(
                CircuitMode.Clifford,
                CliffordRzOptimization(level='light', keep_phase=kp_ph, optimize_toffoli=True)
            )
            self.add_optimizer(
                CircuitMode.CliffordRz,
                CliffordRzOptimization(level='heavy', keep_phase=kp_ph, optimize_toffoli=True)
            )
            self.add_optimizer(
                CircuitMode.Arithmetic,
                TemplateOptimization(template_typelist=[GateType.x, GateType.cx, GateType.ccx]),
            )
            self.add_optimizer(
                CircuitMode.Arithmetic,
                CliffordRzOptimization(level='light', keep_phase=kp_ph, optimize_toffoli=True)
            )
            self.add_optimizer(CircuitMode.Misc, CommutativeOptimization(keep_phase=kp_ph))

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
            # merge a clf gate into a clf+rz block with half probability
            if random.random() > self.clifford_threshold:
                return CircuitMode.CliffordRz
        if mode_prev == CircuitMode.Arithmetic and g.type in [GateType.x, GateType.cx, GateType.ccx]:
            return CircuitMode.Arithmetic
        return None

    def _get_merged_mode(self, prev: SubCircuit, succ: SubCircuit):
        """
        Returns:
            CircuitMode: the mode if `prev` and `succ` is merged
        """

        # Case 1: merge circuits with the same mode
        if prev.mode == succ.mode:
            return prev.mode

        robust_mode = {CircuitMode.CliffordRz, CircuitMode.Misc}
        if min(prev.circuit.size(), succ.circuit.size()) <= self.merge_threshold:
            if (prev.mode, succ.mode) == (CircuitMode.Clifford, CircuitMode.CliffordRz):
                # Case 2.1 [Clf & Clf+Rz]: The smaller Clf is, the more possible to merge
                r = prev.circuit.size() / (prev.circuit.size() + succ.circuit.size())
                if random.random() > r:
                    return CircuitMode.CliffordRz
            elif (succ.mode, prev.mode) == (CircuitMode.Clifford, CircuitMode.CliffordRz):
                # Case 2.2 [Clf+Rz & Clf]: The smaller Clf is, the more likely to merge
                r = succ.circuit.size() / (prev.circuit.size() + succ.circuit.size())
                if random.random() > r:
                    return CircuitMode.CliffordRz

            elif prev.size() >= succ.size() and prev.mode in robust_mode:
                # Case 3.1: merge small succ into robust mode
                return prev.mode
            elif succ.size() >= prev.size() and succ.mode in robust_mode:
                # Case 3.2: merge small prev into robust mode
                return succ.mode

            elif {prev.mode, succ.mode} == {CircuitMode.Clifford, CircuitMode.Arithmetic}:
                # Case 4 [Clf & Arithmetic]: the smaller they are, the more likely to merge
                r = 1 - max(prev.circuit.size(), succ.circuit.size()) / self.merge_threshold
                if random.random() > r:
                    return CircuitMode.CliffordRz
        return None

    def _topo_sort_blocks(self, blocks):
        """
        Sort `blocks` in topological order.
        Returns:
            List[SubCircuit]: a list of blocks in topological order
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
        Check if `src` is reachable from `dest` in `blocks`.

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

    def _merge_blocks(self, blocks):
        """
        Merge blocks in `blocks` if possible.
        """
        ret = blocks[0:1]
        for b in blocks[1:]:
            m = self._get_merged_mode(ret[-1], b)
            if m is not None:
                ret[-1].extend(b.circuit.gates)
                ret[-1].mode = m
            else:
                ret.append(b)
        return ret

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

                # Condition 1: gate type is compatible
                appended_mode = self._get_appended_mode(sub_circ.mode, g)
                if appended_mode is None:
                    continue

                # Condition 2: appending to block i forms no cycle
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

        for _ in range(self.merge_iterations):
            blocks = self._merge_blocks(blocks)

        return blocks

    def _dag_based_partition(self, circuit):
        """
        Partition circuit into blocks based on DAG.

        Returns:
            List[SubCircuit]: a list of blocks
        """
        dag = DAGCircuit(circuit)
        blocks = []
        assigned_blk = [-1] * dag.size
        reachable = np.zeros(shape=(dag.size,), dtype=bool)
        for idx, g in enumerate(dag.gates):
            init_mode = self._get_init_mode(g)
            reachable[: idx] = True
            for prev in reversed(range(idx)):
                g_prev = dag.get_node(prev).gate
                if reachable[prev]:
                    # FIXME better order
                    sub_circ = blocks[assigned_blk[prev]]

                    # Condition 1: gate type is compatible
                    appended_mode = self._get_appended_mode(sub_circ.mode, g)
                    if appended_mode is None:
                        continue

                    # Condition 2: form no cycle
                    flag = True
                    for j in dag.get_node(idx).predecessors:
                        if assigned_blk[prev] != assigned_blk[j] and \
                                self._is_reachable(assigned_blk[prev], assigned_blk[j], blocks):
                            flag = False
                            break

                    if flag:
                        sub_circ.append(g)
                        assigned_blk[idx] = assigned_blk[prev]
                        sub_circ.mode = appended_mode
                        break

                    if not g.commutative(g_prev):
                        reachable[list(dag.all_predecessors(prev))] = False

            if assigned_blk[idx] == -1:
                blocks.append(self.SubCircuit(circuit.width(), init_mode))
                assigned_blk[idx] = len(blocks) - 1
                blocks[assigned_blk[idx]].append(g)

            for j in dag.get_node(idx).predecessors:
                if assigned_blk[idx] != assigned_blk[j]:
                    blocks[assigned_blk[j]].succ.add(assigned_blk[idx])

        # topological sort blocks
        blocks = self._topo_sort_blocks(blocks)

        for _ in range(self.merge_iterations):
            blocks = self._merge_blocks(blocks)

        return blocks

    def partition(self, circuit):
        """
        Partition circuit into blocks.

        Returns:
            List[SubCircuit]: a list of blocks
        """

        if self.partition_method == 'circuit_based':
            return self._circuit_based_partition(circuit)
        elif self.partition_method == 'dag_based':
            return self._dag_based_partition(circuit)

        return []

    @OutputAligner()
    def execute(self, circuit):
        """
        It will first partition the circuit into blocks and then apply optimizers to each block.

        Args:
            circuit(Circuit): the circuit to optimize

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
