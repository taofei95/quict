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
            q_0: |0>───────■──────────────■─────────────────────────────────────────────────
                         ┌─┴──┐         ┌─┴──┐    ┌────────────┐┌────────────┐┌────────────┐
            q_1: |0>─────┤ cx ├─────────┤ cx ├────┤ rx(1.1137) ├┤ ry(5.7981) ├┤ ry(3.0596) ├
                    ┌────┴────┴───┐┌────┴────┴───┐└────────────┘└────────────┘└────────────┘
            q_2: |0>┤ rz(0.60045) ├┤ ry(0.25347) ├────────────────────■─────────────■───────
                    ├─────────────┤└────┬───┬────┘┌────────────┐    ┌─┴──┐        ┌─┴──┐
            q_3: |0>┤ rx(0.13725) ├─────┤ h ├─────┤ rz(1.0267) ├────┤ cx ├────────┤ cx ├────
                    └─────────────┘     └───┘     └────────────┘    └────┘        └────┘
        """
        size = width * 10

        layout_list = layout.edge_list
        error_gate = int(size * (1 / (level * 3)))
        normal_gate = size - error_gate

        qubits_indexes = list(range(width))
        based_2q_gates = int(width * 0.3)
        flow_2q_gates = list(range(based_2q_gates, int(based_2q_gates * 2)))
        cir = Circuit(width)
        curr_gate_size = 0

        def _filter_index_obey_qubits(width, layout_list, two_qubit_gates):
            edges = []
            qubits_indexes = list(range(width))
            for _ in range(two_qubit_gates):
                chosen_layout = random.choice(layout_list)
                l = [chosen_layout.u, chosen_layout.v]
                qubits_indexes = list(set(qubits_indexes) - set(l))
                edges.append(l)

            return edges, qubits_indexes

        for _ in range(normal_gate // width + 2):
            # random choice gate from gateset
            curr_2q_gates = random.choice(flow_2q_gates)
            if curr_gate_size + curr_2q_gates > normal_gate:
                curr_2q_gates = normal_gate - curr_gate_size

            curr_gate_size += curr_2q_gates
            biq_edges, rest_points = _filter_index_obey_qubits(width, layout_list, curr_2q_gates)
            for edge in biq_edges:
                gate = gate_builder(gateset.two_qubit_gate, random_params=True)
                gate | cir(edge)
            curr_1q_gates = min(normal_gate - curr_gate_size, len(rest_points))
            curr_gate_size += curr_1q_gates
            for _ in range(curr_1q_gates):
                single_gate = gate_builder(random.choice(gateset.one_qubit_gates), random_params=True)
                index = random.choice(rest_points)
                single_gate | cir(index)
            if cir.size() < normal_gate:
                continue

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
            q_0: |0>──■─────■─────────────────■─────────────────────────────────────────■───
                    ┌─┴──┐┌─┴──┐            ┌─┴──┐                                    ┌─┴──┐
            q_1: |0>┤ cx ├┤ cx ├──■─────■───┤ cx ├──■─────■───────────■─────■─────■───┤ cx ├
                    └────┘└────┘┌─┴──┐┌─┴──┐└────┘┌─┴──┐┌─┴──┐      ┌─┴──┐┌─┴──┐┌─┴──┐└────┘
            q_2: |0>────────────┤ cx ├┤ cx ├──────┤ cx ├┤ cx ├──■───┤ cx ├┤ cx ├┤ cx ├──────
                                └────┘└────┘      └────┘└────┘┌─┴──┐└────┘└────┘└────┘
            q_3: |0>──────────────────────────────────────────┤ cx ├────────────────────────
                                                              └────┘
        """
        reset_list, normal_list = [], []
        layout_list = layout.edge_list
        for i in layout_list:
            reset_list.append(i.u)
            reset_list.append(i.v)
        a = max(reset_list, key=reset_list.count)  # Find the bits that appear most frequently in the topology
        for j in range(len(layout_list)):
            if layout_list[j].u == a or layout_list[j].v == a:
                normal_list.append(layout_list[j])  # Associated topology nodes for identified qubits
        reset_list = list(set(layout_list) - set(normal_list))  # Associative topological nodes of uncertain qubits

        size = width * 10
        error_gate = int(size * (1 / (level * 3)))

        cir = Circuit(width)
        for _ in range(size - error_gate):
            gate = gate_builder(gateset.two_qubit_gate, random_params=True)
            r_index = np.random.choice(normal_list)
            index = [r_index.u, r_index.v]
            gate | cir(index)

        for i in range(error_gate):
            gate = gate_builder(gateset.two_qubit_gate, random_params=True)
            l_index = np.random.choice(reset_list)
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
            q_0: |0>────────────────────────────────────────────────────────────────────────

            q_1: |0>──■─────────────────────────────■─────■───────────────────────■─────────
                    ┌─┴──┐                        ┌─┴──┐┌─┴──┐                  ┌─┴──┐
            q_2: |0>┤ cx ├──■─────■─────■─────■───┤ cx ├┤ cx ├──■─────■─────■───┤ cx ├──■───
                    └────┘┌─┴──┐┌─┴──┐┌─┴──┐┌─┴──┐└────┘└────┘┌─┴──┐┌─┴──┐┌─┴──┐└────┘┌─┴──┐
            q_3: |0>──────┤ cx ├┤ cx ├┤ cx ├┤ cx ├────────────┤ cx ├┤ cx ├┤ cx ├──────┤ cx ├
                          └────┘└────┘└────┘└────┘            └────┘└────┘└────┘      └────┘
        """
        normal_list = []
        layout_list = layout.edge_list
        level_param = [0.4, 0.2, 0.0]

        cir = Circuit(width)
        gates = width * 10
        qubits_indexes = list(range(width))
        reset_qubits = random.sample(qubits_indexes, int(width * level_param[level - 1]))  # Select a qubit to be in the idle state
        if len(reset_qubits) > 0:
            for q in range(len(reset_qubits)):
                for l in layout_list:
                    if l.u != reset_qubits[q] and l.v != reset_qubits[q]:
                        normal_list.append([l.u, l.v])
        else:
            for l in layout_list:
                normal_list.append([l.u, l.v])

        for _ in range(gates):
            gate = gate_builder(gateset.two_qubit_gate, random_params=True)
            index = random.choice(normal_list)
            gate | cir(index)

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
                    ┌─────────┐   ┌───┐       ┌─┐        ┌─┐       ┌───┐
            q_0: |0>┤ ry(π/2) ├───┤ h ├───────┤M├────────┤M├───────┤ h ├──────────────
                    ├─────────┤┌──┴───┴──┐┌───┴─┴───┐┌───┴─┴───┐┌──┴───┴──┐┌─────────┐
            q_1: |0>┤ rz(π/2) ├┤ rz(π/2) ├┤ ry(π/2) ├┤ rz(π/2) ├┤ rx(π/2) ├┤ rz(π/2) ├
                    ├─────────┤├─────────┤├─────────┤└─────────┘└─────────┘└─────────┘
            q_2: |0>┤ rz(π/2) ├┤ rx(π/2) ├┤ rx(π/2) ├─────────────────────────────────
                    └──┬───┬──┘└───┬─┬───┘├─────────┤    ┌─┐    ┌─────────┐   ┌───┐
            q_3: |0>───┤ h ├───────┤M├────┤ ry(π/2) ├────┤M├────┤ ry(π/2) ├───┤ h ├───
                       └───┘       └─┘    └─────────┘    └─┘    └─────────┘   └───┘
        """
        cir = Circuit(wires=width, topology=layout)
        gates = width * 10
        # build random circuit
        pro_s = 0.8 # the probability of one qubit gate in all circuit
        len_s, len_d = len(gateset.one_qubit_gates), len([gateset.two_qubit_gate])
        prob = [pro_s / len_s] * len_s + [(1 - pro_s) / len_d] * len_d
        cir.random_append(gates - width, gateset.one_qubit_gates + [gateset.two_qubit_gate], probabilities=prob)
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
