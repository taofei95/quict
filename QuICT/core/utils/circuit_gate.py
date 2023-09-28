from __future__ import annotations
from collections import defaultdict
from typing import Union

from .gate_type import GateType


class GateNode:
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
        doc_string = f"Gate Type: {self._gate.type}; \n Qubits Indexes: {self._qindex}"

        return doc_string

    def _get_layer(self):
        return max([p.depth for p in self._parent])

    def assign_depth(self):
        if len(self._parent) == 0:
            self._layer = 1
        else:
            self._layer = max([p.depth for p in self._parent.values()]) + 1

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
        self._first_layer_gates = []
        self._last_layer_gates = {}

        self._initial_property()

    def _initial_property(self):
        # Gates's Property
        self._size = 0
        self._biq_gates_count = 0
        self._siq_gates_count = 0
        self._gate_type_count = defaultdict(int)

    def reset(self):
        self._gates = []
        self._first_layer_gates = []
        self._last_layer_gates = {}

        self._initial_property()

    def copy(self):
        _new = CircuitGates()
        for gate, qidxes in self._gates:
            _new._gates.append((gate, qidxes.copy()))

        # property copy
        _new._size = self.size
        _new._siq_gates_count = self.siq_gates_count
        _new._biq_gates_count = self.biq_gates_count
        _new._gate_type_count = self._gate_type_count.copy()

        return _new

    def append(self, gate, qidxes: list, size: int = 1):
        # Update GateInfo
        self._size += size
        self._analysis_gate(gate)

        # Construct the GateNode
        curr_node = GateNode(gate, qidxes)

        # Find the parents from qidxes
        for q_id in qidxes:
            if q_id in self._last_layer_gates.keys():
                curr_node.assign_parent(q_id, self._last_layer_gates[q_id])
                self._last_layer_gates[q_id].assign_child(q_id, curr_node)

            self._last_layer_gates[q_id] = curr_node

        # Update Depth for curr_node and record GateNode
        curr_node.assign_depth()
        if curr_node.depth == 1:
            self._first_layer_gates.append(curr_node)

        self._gates.append((gate, qidxes))

    def extend(self, gates, qidxes: list = None):
        """ Add a CompositeGate. """
        # Reassign the extend_gates
        extend_gates: CircuitGates = gates._gates.copy()
        extend_gates.reassign(qidxes, self.depth(qidxes))

        # Connectted the self.last_gates with first_gates in circuit_gates
        for initial_node in extend_gates._first_layer_gates:
            for qidx in initial_node.indexes:
                if qidx in self._last_layer_gates.keys():
                    initial_node.assign_parent(qidx, self._last_layer_gates[qidx])

        # Update self._last_gates with circuit_gates
        for qidx, end_node in extend_gates._last_layer_gates.items():
            self._last_layer_gates[qidx] = end_node

        # Update gate properties
        self._size += extend_gates.size
        self._biq_gates_count += extend_gates.biq_gates_count
        self._siq_gates_count += extend_gates.siq_gates_count
        for key, value in extend_gates._gate_type_count.items():
            self._gate_type_count[key] += value

        self._gates.append((gates, qidxes))

    def pop(self, idx: int):
        pop_node = self._gates.pop(idx)
        self._analysis_gate(pop_node.gate, minus=-1)
        self._size -= 1

        return pop_node.gate, pop_node.indexes

    def _analysis_gate(self, gate, minus: int = 1):
        # Gates count update
        if gate.controls + gate.targets == 1:
            self._siq_gates_count += minus
        elif gate.controls + gate.targets == 2:
            self._biq_gates_count += minus

        self._gate_type_count[gate.type] += minus

    def depth(self, indexes: list = None) -> Union[dict, int]:
        """ Return the depth of the circuit.

        Returns:
            list: The Depth for each qubits
        """
        if indexes is not None:
            ds = {}
            for idx in indexes:
                value = self._last_layer_gates[idx] if idx not in self._last_layer_gates.keys() else 0
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

        # Reassign
        update_child = []
        for initial_node in self._first_layer_gates:
            if initial_depth is not None:
                new_layer = max([initial_depth[qubits_mapping[q]] for q in initial_node.indexes])
                initial_node._layer = new_layer

            initial_node._qindex = [qubits_mapping[q] for q in initial_node.indexes]
            for child in initial_node.children:
                if child not in update_child:
                    update_child.append(child)

        while len(update_child) != 0:
            next_layer = []
            for uchild in update_child:
                if initial_depth is not None:
                    uchild.assign_depth()

                uchild._qindex = [qubits_mapping[q] for q in uchild.indexes]
                for nchild in uchild.children:
                    if nchild not in next_layer:
                        next_layer.append(nchild)

            update_child = next_layer

    def LTS(self):
        update_node = self._first_layer_gates
        LST_gates, curr_layer = [], 1
        while len(update_node) > 0:
            next_layer = []
            for node in update_node:
                if node.depth != curr_layer:
                    next_layer.append(node)
                    continue

                LST_gates.append((node._gate, node._qindex))
                node.release()
                for cnode in node.children:
                    if not cnode.occupy:
                        next_layer.append(cnode)
                        cnode.hold()

            update_node = next_layer
            curr_layer += 1

        return LST_gates

    def RTS(self):
        update_node = set(self._last_layer_gates.values())
        RST_gates, curr_layer = [], self.depth()
        while len(update_node) > 0:
            pre_layer = []
            for node in update_node:
                if node.depth != curr_layer:
                    pre_layer.append(node)
                    node.hold()
                    continue

                RST_gates.append((node._gate, node._qindex))
                node.release()
                for pnode in node.parent:
                    if not pnode.occupy:
                        pre_layer.append(pnode)
                        pnode.hold()

            update_node = pre_layer
            curr_layer -= 1

        return RST_gates

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
                self.extend(cgate._gates)
            else:
                self.append(gate, qidxes)
