from __future__ import annotations
from collections import defaultdict
from typing import Union

from .gate_type import GateType


class GateNode:
    @property
    def gate(self):
        return self._gate

    @property
    def depth(self):
        return self._layer

    @property
    def indexes(self):
        return self._qindex

    @property
    def occupy(self) -> bool:
        return self._hold

    @property
    def children(self) -> list:
        return self._children.values()

    @property
    def parent(self) -> list:
        return self._parent.values()

    def __init__(self, gate, qindex: list, parent: dict = None, children: dict = None):
        self._gate = gate
        self._qindex = qindex
        self._parent = parent if parent is not None else {}
        self._children = children if children is not None else {}

        self._layer = 0
        self._hold = False

    def __str__(self):
        doc_string = f"Gate Type: {self._gate.type}; \n Qubits Indexes: {self._qindex}; \n Layer: {self.depth}"

        return doc_string

    def _get_layer(self):
        return max([p.depth for p in self._parent])

    def get_parent(self, index):
        if index not in self._parent.keys():
            return None
        else:
            return self._parent[index]

    def get_child(self, index):
        if index not in self._children.keys():
            return None
        else:
            return self._children[index]

    def assign_depth(self):
        if len(self._parent) == 0:
            self._layer = 1
        else:
            self._layer = max([p.depth for p in self._parent.values()]) + 1

    def assign_qidx(self, qubits_mapping):
        self._qindex = [qubits_mapping[q] for q in self.indexes]
        _parent, _children = {}, {}
        for qidx, par in self._parent.items():
            _parent[qubits_mapping[qidx]] = par

        for qidx, child in self._children.items():
            _children[qubits_mapping[qidx]] = child

        self._parent = _parent
        self._children = _children

    def assign_child(self, qidx, gate):
        self._children[qidx] = gate

    def assign_parent(self, qidx, gate):
        self._parent[qidx] = gate

    def copy(self):
        _new = GateNode(
            self._gate, self._qindex.copy(),
            self._parent.copy(), self._children.copy()
        )

        _new._layer = self._layer

        return _new

    def hold(self):
        self._hold = True

    def release(self):
        self._hold = False


class CircuitGates:
    @property
    def gates(self) -> list:
        return self._gates

    @property
    def size(self) -> int:
        return self._size

    @property
    def qubits(self) -> list:
        return sorted(self._last_layer_gates.keys())

    @property
    def length(self) -> int:
        return len(self._gates)

    @property
    def width(self) -> int:
        return len(self._last_layer_gates.keys())

    @property
    def siq_gates_count(self) -> int:
        return self._siq_gates_count

    @property
    def biq_gates_count(self) -> int:
        return self._biq_gates_count

    def gates_count_by_type(self, gate_type: GateType) -> int:
        return self._gate_type_count[gate_type]

    @property
    def training_gates_count(self) -> int:
        return 0

    def __init__(self):
        self._gates = []
        self._first_layer_gates = {}
        self._last_layer_gates = {}

        self._initial_property()

    def __call__(self):
        return self._gates

    def _initial_property(self):
        # Gates's Property
        self._size = 0
        self._biq_gates_count = 0
        self._siq_gates_count = 0
        self._gate_type_count = defaultdict(int)

    def reset(self):
        self._gates = []
        self._first_layer_gates = {}
        self._last_layer_gates = {}

        self._initial_property()

    def copy(self):
        _new = CircuitGates()
        for gate in self._gates:
            _new._gates.append(gate.copy())

        _new._first_layer_gates = self._first_layer_gates.copy()
        _new._last_layer_gates = self._last_layer_gates.copy()

        # property copy
        _new._size = self.size
        _new._siq_gates_count = self.siq_gates_count
        _new._biq_gates_count = self.biq_gates_count
        _new._gate_type_count = self._gate_type_count.copy()

        return _new

    def append(self, gate, qidxes: list):
        # Update GateInfo
        self._analysis_gate(gate)

        # Construct the GateNode
        curr_node = GateNode(gate.copy(), qidxes)

        # Find the parents from qidxes
        for q_id in qidxes:
            if q_id in self._last_layer_gates.keys():
                curr_node.assign_parent(q_id, self._last_layer_gates[q_id])
                self._last_layer_gates[q_id].assign_child(q_id, curr_node)
            else:
                self._first_layer_gates[q_id] = curr_node

            self._last_layer_gates[q_id] = curr_node

        # Update Depth for curr_node and record GateNode
        curr_node.assign_depth()
        self._gates.append(curr_node)

    def extend(self, gates, qidxes: list):
        """ Add a CompositeGate. """
        # Reassign the extend_gates
        cgate_copy = gates.copy()
        extend_gates: CircuitGates = cgate_copy._gates
        extend_gates.reassign(qidxes, self.depth(qidxes))

        # Connectted the self.last_gates with first_gates in circuit_gates
        for key, initial_node in extend_gates._first_layer_gates.items():
            for qidx in initial_node.indexes:
                if qidx not in initial_node._parent.keys():
                    if qidx in self._last_layer_gates.keys():
                        initial_node.assign_parent(qidx, self._last_layer_gates[qidx])
                        self._last_layer_gates[qidx].assign_child(qidx, initial_node)
                    else:
                        self._first_layer_gates[qidx] = initial_node

        # Update self._last_gates with circuit_gates
        for qidx, end_node in extend_gates._last_layer_gates.items():
            self._last_layer_gates[qidx] = end_node

        # Update gate properties
        self._analysis_compositegate(cgate_copy)
        self._gates.append(cgate_copy)

    def insert_gate(self, gate, qidxes: list, depth: int = None, is_adjust: bool = False):
        # Initial Node
        inode = GateNode(gate, qidxes) if not is_adjust else gate

        # Get connectted nodes by qidxes and depth
        for idx in qidxes:
            if depth == 1:
                iparent, ichild = None, self._first_layer_gates[idx]
            elif depth > self._last_layer_gates[idx].depth:
                iparent, ichild = self._last_layer_gates[idx], None
            else:
                start_node = self._first_layer_gates[idx]
                while start_node.depth < depth:
                    start_node = start_node.get_child(idx)

                iparent = start_node.get_parent(idx)
                ichild = start_node

            if iparent is not None:
                inode.assign_parent(idx, iparent)
                iparent.assign_child(idx, inode)
            else:
                self._first_layer_gates[idx] = inode

            if ichild is not None:
                inode.assign_child(idx, ichild)
                ichild.assign_parent(idx, inode)
            else:
                self._last_layer_gates[idx] = inode

        self._depth_update([inode])
        if not is_adjust:
            self._analysis_gate(gate)
            self._gates.append(inode)
        else:
            return inode

    def insert_cgate(self, gate, qidxes: list, depth: int = None):
        # Initial CompositeGate
        gate_list: CircuitGates = gate._gates
        gate_list.reassign(qidxes)
        update_node = []
        for node in gate_list._first_layer_gates.values():
            if node.depth == 1:
                update_node.append(node)

        # Get connectted nodes by qidxes and depth
        for idx in qidxes:
            if depth == 1:
                iparent, ichild = None, self._first_layer_gates[idx]
            elif depth > self._last_layer_gates[idx].depth:
                iparent, ichild = self._last_layer_gates[idx], None
            else:
                start_node = self._first_layer_gates[idx]
                while start_node.depth < depth:
                    start_node = start_node.get_child(idx)

                iparent = start_node.get_parent(idx)
                ichild = start_node

            start_node, end_node = gate_list._first_layer_gates[idx], gate_list._last_layer_gates[idx]
            if iparent is not None:
                start_node.assign_parent(idx, iparent)
                iparent.assign_child(idx, start_node)
            else:
                self._first_layer_gates[idx] = start_node

            if ichild is not None:
                end_node.assign_child(idx, ichild)
                ichild.assign_parent(idx, end_node)
            else:
                self._last_layer_gates[idx] = end_node

        self._depth_update(update_node)
        self._analysis_compositegate(gate)
        self._gates.append(gate)

    def pop(self, index: int):
        pop_node = self._gates.pop(index)

        if isinstance(pop_node, GateNode):
            return self._pop_node(pop_node)
        else:
            return self._pop_cgate(pop_node)

    def _pop_node(self, node):
        # Update the circuit's property
        self._analysis_gate(node.gate, minus=-1)

        update_node = []
        for idx in node.indexes:
            parent = node.get_parent(idx)
            child = node.get_child(idx)

            if parent is not None:
                if child is not None:
                    parent.assign_child(idx, child)
                    child.assign_parent(idx, parent)
                    update_node.append(child)
                else:
                    del parent._children[idx]
                    self._last_layer_gates[idx] = parent
            else:
                if child is not None:
                    del child._parent[idx]
                    update_node.append(child)
                    self._first_layer_gates[idx] = child
                else:
                    del self._first_layer_gates[idx]
                    del self._last_layer_gates[idx]

        self._depth_update(update_node)

        return node.gate & node.indexes

    def _pop_cgate(self, cgate):
        # Update the circuit's property
        self._analysis_compositegate(cgate, minus=-1)
        gate_list: CircuitGates = cgate._gates

        update_node = []
        for idx in gate_list.qubits:
            first_layer_node: GateNode = gate_list._first_layer_gates[idx]
            last_layer_node: GateNode = gate_list._last_layer_gates[idx]
            parent = first_layer_node.get_parent(idx)
            child = last_layer_node.get_child(idx)

            if parent is not None:
                if child is not None:
                    parent.assign_child(idx, child)
                    child.assign_parent(idx, parent)
                    update_node.append(child)
                else:
                    del parent._children[idx]
                    self._last_layer_gates[idx] = parent
            else:
                if child is not None:
                    del child._parent[idx]
                    update_node.append(child)
                    self._first_layer_gates[idx] = child
                else:
                    del self._first_layer_gates[idx]
                    del self._last_layer_gates[idx]

        self._depth_update(update_node)
        gate_list.restore()
        return cgate

    def adjust(self, index: int, qidxes: list, is_adjust: bool = False, depth: int = 10):
        adj_node = self._gates[index]
        if isinstance(adj_node, GateNode):
            if not is_adjust and qidxes == adj_node.indexes and depth is None:
                return

            self._pop_node(adj_node)
            adj_node = self.insert_gate(adj_node, qidxes, depth, True)
        else:
            if qidxes == adj_node.qubits and depth is None:
                return

            self._pop_cgate(adj_node)
            adj_node = self.insert_cgate(adj_node, qidxes, depth)

        self._gates[index] = adj_node

    def _analysis_gate(self, gate, minus: int = 1):
        # Gates count update
        self._size += minus
        if gate.controls + gate.targets == 1:
            self._siq_gates_count += minus
        elif gate.controls + gate.targets == 2:
            self._biq_gates_count += minus

        self._gate_type_count[gate.type] += minus

    def _analysis_compositegate(self, cgate, minus: int = 1):
        self._size += cgate.size() * minus
        self._biq_gates_count += cgate.count_2qubit_gate() * minus
        self._siq_gates_count += cgate.count_1qubit_gate() * minus
        for key, value in cgate._gates._gate_type_count.items():
            self._gate_type_count[key] += value * minus

    def depth(self, indexes: list = None) -> Union[dict, int]:
        """ Return the depth of the circuit.

        Returns:
            list: The Depth for each qubits
        """
        if indexes is not None:
            ds = {}
            for idx in indexes:
                value = self._last_layer_gates[idx].depth if idx in self._last_layer_gates.keys() else 0
                ds[idx] = value

            return ds

        ds = [node.depth for node in self._last_layer_gates.values()]
        return max(ds)

    def reassign(self, qubit_indexes: list, initial_depth: dict = None):
        # Validation
        assert len(qubit_indexes) == len(self.qubits), \
            "Cannot reassign the CGate/Circuit into the wrong qubits' number."
        if initial_depth is not None:
            assert len(initial_depth.keys()) == len(self.qubits), "The length of new initial depth is wrong."

        # Construct qubits mapping
        qubits_mapping = {}
        for i, q in enumerate(self.qubits):
            qubits_mapping[q] = qubit_indexes[i]

        # adjust the depth
        adjust_nodes = self.LTS(node_only=True)
        if initial_depth is not None:
            for _, initial_node in self._first_layer_gates.items():
                if initial_node.depth == 1:
                    new_layer = max([initial_depth[qubits_mapping[q]] for q in initial_node.indexes])
                    initial_node._layer = new_layer + 1

        # Reassign the qubit indexes for gates
        for node in adjust_nodes:
            node.assign_qidx(qubits_mapping)
            if len(node.parent) == 0:
                continue

            if initial_depth is not None:
                node.assign_depth()

        # Update the first layer and the last layer
        _first_layer_gates, _last_layer_gates = {}, {}
        for key, node in self._first_layer_gates.items():
            _first_layer_gates[qubits_mapping[key]] = node

        for key, node in self._last_layer_gates.items():
            _last_layer_gates[qubits_mapping[key]] = node

        self._first_layer_gates = _first_layer_gates
        self._last_layer_gates = _last_layer_gates

    def restore(self):
        start_node = []
        for idx, node in self._first_layer_gates.items():
            del node._parent[idx]
            if len(node.parent) == 0:
                start_node.append(node)

        for idx, node in self._last_layer_gates.items():
            del node._children[idx]

        self._depth_update(start_node)

    def LTS(self, node_only: bool = False):
        update_node = set(self._first_layer_gates.values())
        for node in update_node:
            node.hold()

        LST_gates, curr_layer = [], 1
        while len(update_node) > 0:
            next_layer = []
            for node in update_node:
                if node.depth != curr_layer:
                    next_layer.append(node)
                    continue

                if node_only:
                    LST_gates.append(node)
                else:
                    LST_gates.append((node._gate, node._qindex))

                node.release()
                for cnode in node.children:
                    if not cnode.occupy:
                        next_layer.append(cnode)
                        cnode.hold()

            update_node = next_layer
            curr_layer += 1

        return LST_gates

    def RTS(self, node_only: bool = False):
        update_node = set(self._last_layer_gates.values())
        for node in update_node:
            node.hold()

        RST_gates, curr_layer = [], self.depth()
        while len(update_node) > 0:
            pre_layer = []
            for node in update_node:
                if node.depth != curr_layer:
                    pre_layer.append(node)
                    node.hold()
                    continue

                if node_only:
                    RST_gates.append(node)
                else:
                    RST_gates.append((node._gate, node._qindex))

                node.release()
                for pnode in node.parent:
                    if not pnode.occupy:
                        pre_layer.append(pnode)
                        pnode.hold()

            update_node = pre_layer
            curr_layer -= 1

        return RST_gates

    def _depth_update(self, start_node: list):
        for node in start_node:
            node.hold()

        while len(start_node) > 0:
            next_layer = []
            for node in start_node:
                node.assign_depth()
                for cnode in node.children:
                    if not cnode.occupy:
                        next_layer.append(cnode)
                        cnode.hold()

                node.release()

            start_node = next_layer

    def flatten(self):
        # TODO: add qubit mapping change
        flat_gates = []
        for gate, _ in self._gates:
            if type(gate).__name__ == "CompositeGate":
                gate.flatten()
                flat_gates.extend(gate._gates)
            else:
                flat_gates.append(gate)

        self._gates = flat_gates

    def decomposition(self):
        decomp_gates = self.LTS()
        self.reset()
        for gate, qidxes in decomp_gates:
            cgate = gate.build_gate(qidxes)
            if cgate is not None:
                self.extend(cgate)
            else:
                self.append(gate, qidxes)
