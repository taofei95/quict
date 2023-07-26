import random
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.layout.layout import Layout
from QuICT.core.virtual_machine.instruction_set import InstructionSet


class BenchmarkCircuitBuilder:
    """
    A class fetch special benchmark circuits.

    Args:
        width(int): number of qubits.
        level(int): level of benchmark circuits.
        gateset(InstructionSet): The set of quantum gates which Quantum Machine supports.
        layout (Layout, optional): The description of physical topology of Quantum Machine.
    """
    @staticmethod
    def parallelized_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        """
        Highly parallel applications place a large number of operations into a relatively small circuit depth.
        Example:
                    ┌────────────┐ ┌────────────┐
            q_0: |0>┤ ry(5.4598) ├─┤ rz(5.9459) ├────────────────────────────────────────────────────■─────────■─────────
                    └───┬───┬────┘┌┴────────────┴┐┌────────────┐┌────────────┐                       │       ┌─┴──┐
            q_1: |0>────┤ h ├─────┤ ry(0.068788) ├┤ rx(4.5208) ├┤ rx(1.0925) ├─────────────■─────────┼───────┤ cx ├──────
                    ┌───┴───┴────┐└──────────────┘├────────────┤├────────────┤             │       ┌─┴──┐    └────┘
            q_2: |0>┤ rx(5.4171) ├───────■────────┤ rz(5.9046) ├┤ ry(5.6318) ├──■──────────┼───────┤ cx ├──────■─────■───
                    ├────────────┤     ┌─┴──┐     └───┬───┬────┘└────────────┘┌─┴──┐┌───┐┌─┴──┐┌───┴────┴───┐┌─┴──┐┌─┴──┐
            q_3: |0>┤ ry(1.5532) ├─────┤ cx ├─────────┤ h ├───────────────────┤ cx ├┤ h ├┤ cx ├┤ rx(4.7973) ├┤ cx ├┤ cx ├
                    └────────────┘     └────┘         └───┘                   └────┘└───┘└────┘└────────────┘└────┘└────┘
        """
        size = width * 10

        layout_list = layout.edge_list
        error_gate = int(size * (1 / (level * 3)))
        normal_gate = size - error_gate

        qubits_indexes = list(range(width))
        based_2q_gates = int(width * 0.2)
        flow_2q_gates = list(range(based_2q_gates // 2, int(based_2q_gates * 1.5)))

        cir = Circuit(width)
        curr_gate_size = 0
        for _ in range(normal_gate // width + 1):
            # random choice gate from gateset
            curr_2q_gates = random.choice(flow_2q_gates)
            if curr_gate_size + curr_2q_gates > normal_gate:
                curr_2q_gates = normal_gate - curr_gate_size

            curr_gate_size += curr_2q_gates
            # TODO: Function: input(qubits, 2q-gates, edges) -> edges, points
            biq_edges = [[1, 2]]
            for edge in biq_edges:
                gate = gate_builder(gateset.two_qubit_gate, random_params=True)
                gate | cir(edge)

            rest_points = set(qubits_indexes)
            curr_1q_gates = min(normal_gate - curr_gate_size, len(rest_points))
            curr_gate_size += curr_1q_gates
            for i in range(curr_1q_gates):
                single_gate = gate_builder(random.choice(gateset.one_qubit_gates), random_params=True)
                single_gate | cir(rest_points[i])

        for _ in range(error_gate):
            insert_index = random.choice(list(range(size)))
            typelist = [random.choice(gateset.one_qubit_gates), gateset.two_qubit_gate]
            rand_idx = np.random.randint(0, 2)
            gate = gate_builder(typelist[rand_idx], random_params=True)
            if gate.is_single():
                bit_point = np.random.randint(0, width)
                cir.insert(gate & bit_point, insert_index)
            else:
                inset_index = np.random.choice(layout_list)
                cir.insert(gate & [inset_index.u, inset_index.v], insert_index)

        depth = cir.depth()
        cir.name = "+".join(
            ["benchmark", "highly_parallelized", f"w{width}_s{size}_d{depth}_level{level}"]
        )

        return cir

    @staticmethod
    def serialized_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        """
        The number of two quantum bits interacting on the longest path of the circuit depth is close to the total number
            of doublets.
        Example:
            q_0: |0>──■─────■─────■─────■─────■─────■─────■─────■─────■───────────■─────■─────■─────■───────────■─────■─────■─────■─────■───
                    ┌─┴──┐  │   ┌─┴──┐  │     │     │   ┌─┴──┐┌─┴──┐┌─┴──┐        │     │     │     │           │   ┌─┴──┐  │     │     │
            q_1: |0>┤ cx ├──┼───┤ cx ├──┼─────┼─────┼───┤ cx ├┤ cx ├┤ cx ├──■─────┼─────┼─────┼─────┼───────────┼───┤ cx ├──┼─────┼─────┼───
                    └────┘┌─┴──┐└────┘┌─┴──┐┌─┴──┐┌─┴──┐└────┘└────┘└────┘  │   ┌─┴──┐┌─┴──┐┌─┴──┐┌─┴──┐      ┌─┴──┐└────┘┌─┴──┐┌─┴──┐┌─┴──┐
            q_2: |0>──────┤ cx ├──────┤ cx ├┤ cx ├┤ cx ├────────────────────┼───┤ cx ├┤ cx ├┤ cx ├┤ cx ├──■───┤ cx ├──────┤ cx ├┤ cx ├┤ cx ├
                          └────┘      └────┘└────┘└────┘                  ┌─┴──┐└────┘└────┘└────┘└────┘┌─┴──┐└────┘      └────┘└────┘└────┘
            q_3: |0>──────────────────────────────────────────────────────┤ cx ├────────────────────────┤ cx ├──────────────────────────────
                                                                          └────┘                        └────┘
        """
        l_list, r_list = [], []
        layout_list = layout.edge_list
        for i in layout_list:
            l_list.append(i.u)
            l_list.append(i.v)
        a = max(l_list, key=l_list.count)  # Find the bits that appear most frequently in the topology
        for j in range(len(layout_list)):
            if layout_list[j].u == a or layout_list[j].v == a:
                r_list.append(layout_list[j])  # Associated topology nodes for identified qubits
        l_list = list(set(layout_list) - set(r_list))  # Associative topological nodes of uncertain qubits

        size = width * 10
        error_gate = int(size * (1 / (level * 3)))

        cir = Circuit(width)
        for _ in range(size - error_gate):
            gate = gate_builder(gateset.two_qubit_gate, random_params=True)
            r_index = np.random.choice(r_list)
            index = [r_index.u, r_index.v]
            gate | cir(index)

        for i in range(error_gate):
            gate = gate_builder(gateset.two_qubit_gate, random_params=True)
            l_index = np.random.choice(l_list)
            index = [l_index.u, l_index.v]
            insert_index = np.random.randint(0, size - error_gate + i)
            cir.insert(gate & index, insert_index)

        depth = cir.depth()
        cir.name = "+".join(
            ["benchmark", "highly_serialized", f"w{width}_s{size}_d{depth}_level{level}"]
        )

        return cir

    @staticmethod
    def entangled_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        """
        By calculating the two quantum bits interacting to a maximum value.
        Example:
            q_0: |0>──■─────■─────■─────■─────■─────■─────■─────■─────────────────■─────■─────■─────■───
                    ┌─┴──┐┌─┴──┐  │   ┌─┴──┐  │   ┌─┴──┐┌─┴──┐┌─┴──┐            ┌─┴──┐┌─┴──┐┌─┴──┐  │
            q_1: |0>┤ cx ├┤ cx ├──┼───┤ cx ├──┼───┤ cx ├┤ cx ├┤ cx ├────────■───┤ cx ├┤ cx ├┤ cx ├──┼───
                    └────┘└────┘┌─┴──┐└────┘┌─┴──┐└────┘└────┘└────┘        │   └────┘└────┘└────┘┌─┴──┐
            q_2: |0>────────────┤ cx ├──■───┤ cx ├──■─────■─────■─────■─────┼─────■─────■─────────┤ cx ├
                                └────┘┌─┴──┐└────┘┌─┴──┐┌─┴──┐┌─┴──┐┌─┴──┐┌─┴──┐┌─┴──┐┌─┴──┐      └────┘
            q_3: |0>──────────────────┤ cx ├──────┤ cx ├┤ cx ├┤ cx ├┤ cx ├┤ cx ├┤ cx ├┤ cx ├────────────
                                      └────┘      └────┘└────┘└────┘└────┘└────┘└────┘└────┘
        """
        qubits_indexes = list(range(width))
        gate_2q = gate_builder(gateset.two_qubit_gate, random_params=True)
        layout_list = layout.edge_list
        level_param = [0.4, 0.2, 0.0]

        cir = Circuit(width)
        gates = width * 10
        while cir.size() < gates:
            l_list = []
            a = random.choice(qubits_indexes)  # Select a qubit to be in the idle state
            for _ in range(int(width * level_param[level - 1])):
                for j in range(len(layout_list)):
                    if layout_list[j].u == a or layout_list[j].v == a:
                        l_list.append(layout_list[j])
                        # Removing idle qubit-related topologies from the list
                        r_list = list(set(layout_list) - set(l_list))
            if len(l_list) != 0:
                layout_list = r_list

            for _ in range(len(layout_list)):
                if cir.size() < gates:
                    r_index = np.random.choice(layout_list)
                    index = [r_index.u, r_index.v]
                    insert_index = random.choice(list(range(width)))
                    cir.insert(gate_2q & index, insert_index)
            depth = cir.depth()
            cir.name = "+".join(
                ["benchmark", "highly_entangled", f"w{width}_s{gates}_d{depth}_level{level}"]
            )

        return cir

    @staticmethod
    def mediate_measure_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        """
        For circuits consisting of multiple consecutive layers of gate operations, the measurement gates extract informa
            tion at different layers for the duration of and after the execution of the programme.
        Example:
                                                        ┌───┐       ┌─┐        ┌─┐    ┌─────────┐
            q_0: |0>───────────────────────────■────────┤ h ├───────┤M├────────┤M├────┤ rz(π/2) ├
                    ┌─────────┐                │        └┬─┬┘   ┌───┴─┴───┐    └─┘    └─────────┘
            q_1: |0>┤ ry(π/2) ├────────────────┼─────────┤M├────┤ rx(π/2) ├──────────────────────
                    └─────────┘┌─────────┐   ┌─┴──┐     ┌┴─┴┐   └───┬─┬───┘┌─────────┐┌─────────┐
            q_2: |0>─────■─────┤ ry(π/2) ├───┤ cx ├─────┤ h ├───────┤M├────┤ rz(π/2) ├┤ rx(π/2) ├
                       ┌─┴──┐  └──┬───┬──┘┌──┴────┴─┐┌──┴───┴──┐┌───┴─┴───┐├─────────┤├─────────┤
            q_3: |0>───┤ cx ├─────┤ h ├───┤ rz(π/2) ├┤ rz(π/2) ├┤ ry(π/2) ├┤ rx(π/2) ├┤ ry(π/2) ├
                       └────┘     └───┘   └─────────┘└─────────┘└─────────┘└─────────┘└─────────┘
        """
        layout_list = layout.edge_list
        pro_s = 1 - level / 10

        cir = Circuit(width)
        gates = width * 10
        while cir.size() < gates - width:
            # Single-qubit gates
            size_s = int(gates * pro_s)
            cir.random_append(size_s, gateset.one_qubit_gates)
            # Double-qubits gates
            size_d = gates - width - size_s
            for _ in range(size_d):
                biq_gate = gate_builder(gateset.two_qubit_gate, random_params=True)
                bgate_layout = np.random.choice(layout_list)
                insert_idx = random.choice(list(range(width)))
                cir.insert(biq_gate & [bgate_layout.u, bgate_layout.v], insert_idx)
        # insert measure to circuit
        for _ in range(width):
            index = random.choice(list(range(width)))
            mea_index = random.choice(list(range(int(gates * 0.25), int(gates * 0.75))))
            cir.insert(Measure & index, mea_index)

        depth = cir.depth()
        cir.name = "+".join(
            ["benchmark", "mediate_measure", f"w{width}_s{gates}_d{depth}_level{level}"]
        )

        return cir

    def get_benchmark_circuit(
        self,
        width: int,
        level: int,
        gateset: InstructionSet,
        layout: Layout,
        N: int = 4
    ):
        """get all special benchmark circuits

        Args:
            width(int): number of qubits.
            level(int): level of benchmark circuits.
            gateset(InstructionSet): The set of quantum gates which Quantum Machine supports.
            layout (Layout, optional): The description of physical topology of Quantum Machine.
            N (int, optional): The number of circuit repetitions. Defaults to 4.

        """
        cirs_list = []
        for _ in range(N):
            cirs_list.append(self.parallelized_circuit_build(width, level, gateset, layout))
            cirs_list.append(self.entangled_circuit_build(width, level, gateset, layout))
            cirs_list.append(self.serialized_circuit_build(width, level, gateset, layout))
            cirs_list.append(self.mediate_measure_circuit_build(width, level, gateset, layout))

        return cirs_list
