import numpy as np
from QuICT.core import layout
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate.gate import gate_builder
from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType


class QuantumMachineCircuitBuilder:
    " A class fetch quantum machine circuits."

    _machine_instructionset_list = {
        "origin": [[GateType.u3], [GateType.cz]],
        "ibm": [[GateType.id, GateType.rz, GateType.sx, GateType.x], [GateType.cx]],
        "quafu": [[GateType.h, GateType.rx, GateType.ry, GateType.rz], [GateType.cx]],
        "baidu": [[GateType.u3], [GateType.cz]],
        "guodun": [[GateType.x, GateType.y, GateType.z], [GateType.cz]]
    }

    def _ibm_layout(self):
        ibm_layout_list = []
        # ibm machine
        ibm_layout_list.append(Layout.linear_layout(5))
        ibm_layout_list.append(Layout.linear_layout(12))
        edges_list = [
            [[0, 1], [1, 3], [2, 1], [3, 4], [4, 3]],
            [[0, 1], [1, 3], [2, 1], [3, 5], [4, 5], [5, 6], [6, 5]],
            [[0, 1], [1, 0], [2, 3], [3, 2], [4, 7], [5, 3], [6, 7], [7, 4], [8, 11], [9, 8], [10, 12], [11, 8],
            [12, 15], [13, 14], [14, 13], [15, 12]],
            [[0, 1], [1, 4], [1, 2], [2, 3], [3, 5], [4, 7], [5, 8], [6, 7], [7, 10], [8, 11], [8, 9], [10, 12],
            [11, 14], [12, 15], [12, 13], [13, 14], [15, 12], [15, 18], [19, 16], [19, 20], [19, 22], [21, 18],
            [21, 23], [23, 27], [24, 25], [24, 23], [25, 22]],
            [[0, 1], [1, 4], [1, 2], [2, 3], [3, 5], [4, 7], [5, 8], [6, 7], [7, 10], [8, 11], [8, 9], [10, 12],
            [11, 14], [12, 15], [12, 13], [13, 14], [15, 12], [15, 18], [19, 16], [19, 20], [19, 22], [21, 18],
            [21, 23], [23, 27], [24, 25], [24, 23], [25, 22], [25, 26], [27, 28], [28, 29], [30, 3], [30, 31], [31, 32]]
        ]
        for edges in edges_list:
            layout_5q = Layout(5)
            for i in range(len(edges)):
                layout_5q.add_edge(edges[i][0], edges[i][1])
                ibm_layout_list.append(layout_5q)

        return ibm_layout_list

    def _guodun_layout(self):
        guodun_layout_list = []

        unreachable_nodes = [2, 18, 22, 27, 37, 42, 54, 60, 61, 62, 66]
        layout_66 = Layout.grid_layout(qubit_number=66, unreachable_nodes=unreachable_nodes)
        guodun_layout_list.append(layout_66)
        guodun_layout_list.append(Layout.linear_layout(12))
        return guodun_layout_list

    def _get_machine_topology(self):
        ibm_layout = self._ibm_layout()
        machine_topology = {
            "origin": [Layout.linear_layout(6)],
            "ibm": self._ibm_layout(),
            "quafu": [Layout.linear_layout(8), Layout.linear_layout(18)],
            "baidu": [Layout.linear_layout(10)],
            "guodun": self._guodun_layout()
        }
        return machine_topology

    def get_machine_cir(self):
        quantum_machine_cir = []
        gate_number = [5, 10, 20, 50]
        machine_type = ["origin", "ibm", "quafu", "baidu", "guodun"]

        for type in machine_type:
            gate_type = self._machine_instructionset_list[type]
            machine_topology = self._get_machine_topology()
            layout_list = machine_topology[type]
            for layout in layout_list:
                q = layout.qubit_number
                for g in gate_number:
                    cir = Circuit(q)
                    # Single-qubit gates
                    size_s = int(g * q * 0.8)
                    cir.random_append(size_s, gate_type[0])
                    # Double-qubits gates
                    size_d = g - size_s
                    layout_list = layout.edge_list
                    for _ in range(size_d):
                        biq_gate = gate_builder(gate_type[1], random_params=True)
                        bgate_layout = np.random.choice(layout_list)
                        biq_gate | cir([bgate_layout.u, bgate_layout.v])
                    cir.name = "+".join([type, "machine", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
                    quantum_machine_cir.append(cir)

        return quantum_machine_cir
