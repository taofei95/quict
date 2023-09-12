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
                         ┌───┐     ┌────────────┐┌─────────────┐
            q_0: |0>─────┤ h ├─────┤ rx(1.0595) ├┤ rz(0.54992) ├──────■─────────────────────
                    ┌────┴───┴────┐├────────────┤└┬────────────┤    ┌─┴──┐
            q_1: |0>┤ ry(0.65449) ├┤ rz(1.1139) ├─┤ rz(2.0863) ├────┤ cx ├──────────────────
                    └┬────────────┤├────────────┤ ├────────────┤┌───┴────┴───┐
            q_2: |0>─┤ ry(3.5725) ├┤ ry(1.7149) ├─┤ rx(2.2274) ├┤ ry(3.6362) ├──────────────
                     ├────────────┤└───┬───┬────┘ ├────────────┤├────────────┤┌────────────┐
            q_3: |0>─┤ ry(4.0293) ├────┤ h ├──────┤ rz(4.6169) ├┤ rz(4.6137) ├┤ ry(5.0187) ├
                     └────────────┘    └───┘      └────────────┘└────────────┘└────────────┘
        """
        def _filter_index_obey_qubits(width, layout_list, two_qubit_gates):
            # choose the layout of two qubits, resurn layout edges and reset qubits
            edges = layout_list.copy()
            qubits_indexes, choice_edges = list(range(width)), []
            for _ in range(two_qubit_gates):
                chosen_layout = random.choice(edges)
                l = [chosen_layout.u, chosen_layout.v]
                choice_edges.append(l)

                new_edges = []
                for edge in edges:
                    if edge.u not in l and edge.v not in l:
                        new_edges.append(edge)

                for qidx in l:
                    qubits_indexes.remove(qidx)

                if len(new_edges) == 0:
                    break

                edges = new_edges[:]

            return choice_edges, qubits_indexes

        # Base information
        size = width * 10
        parallel_layers = 3 * level
        normal_gate_size = (10 - parallel_layers) * width
        layout_list = layout.edge_list
        based_2q_gates = int(width * 0.2)

        # Build Circuit
        cir = Circuit(width, topology=layout)
        for i in range(parallel_layers):
            # append two qubits gate into this layer
            curr_2q_gates = np.random.randint(based_2q_gates // 2, int(based_2q_gates * 1.5) + 1)
            biq_edges, rest_points = _filter_index_obey_qubits(width, layout_list, curr_2q_gates)
            # insert two qubits gate obey chosen layout
            for edge in biq_edges:
                gate = gate_builder(gateset.two_qubit_gate, random_params=True)
                gate | cir(edge)

            # insert one qubit gate obey reset qubits
            for sp in rest_points:
                single_gate = gate_builder(random.choice(gateset.one_qubit_gates), random_params=True)
                single_gate | cir(sp)

        # Random append the rest layers.
        pro_s = 0.8  # the probability of one qubit gate in all circuit
        len_s, len_d = len(gateset.one_qubit_gates), len([gateset.two_qubit_gate])
        prob = [pro_s / len_s] * len_s + [(1 - pro_s) / len_d] * len_d
        cir.random_append(
            normal_gate_size,
            gateset.one_qubit_gates + [gateset.two_qubit_gate],
            probabilities=prob,
            random_params=True
        )

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
            if len(reset_list) > 0:
                l_index = random.choice(reset_list)
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
                    ┌───┐ ┌─────────────┐    ┌───┐
            q_0: |0>┤ h ├─┤ rx(0.82176) ├────┤ h ├────────────────────────────────────────────────
                    └───┘ └────┬───┬────┘┌───┴───┴────┐┌────────────┐
            q_1: |0>──■────────┤ h ├─────┤ ry(4.0277) ├┤ ry(2.8401) ├─────────────────────────────
                    ┌─┴──┐ ┌───┴───┴────┐└────────────┘├────────────┤┌─────────────┐
            q_2: |0>┤ cx ├─┤ rx(1.1725) ├──────■───────┤ rz(3.4419) ├┤ rx(0.44925) ├──────────────
                    ├───┬┘ └───┬───┬────┘    ┌─┴──┐    ├────────────┤└┬────────────┤┌────────────┐
            q_3: |0>┤ h ├──────┤ h ├─────────┤ cx ├────┤ rx(5.9256) ├─┤ rx(5.7601) ├┤ ry(4.1335) ├
                    └───┘      └───┘         └────┘    └────────────┘ └────────────┘└────────────┘
        """
        # Based Information
        level_param = [0.6, 0.8, 1.0]
        rest_qubits_number = width - int(width * level_param[level - 1])

        # Divided qubits into two parts
        normal_qubits, rest_qubits = list(range(width)), []
        if rest_qubits_number > 0:
            for _ in range(width):
                a = random.choice(normal_qubits)
                target_qubits = [edge.v for edge in layout.out_edges(a)]
                if len(target_qubits) != 0:
                    break
            rest_qubits.append(a)
            normal_qubits.remove(a)
            for _ in range(rest_qubits_number - 1):
                a = random.choice(target_qubits)
                rest_qubits.append(a)
                target_qubits.remove(a)
                normal_qubits.remove(a)
                for edge in layout.out_edges(a):
                    if edge.v not in rest_qubits and edge.v not in target_qubits:
                        target_qubits.append(edge.v)

                if len(target_qubits) == 0:
                    break

        # Build Circuit
        new_topo = Layout(width)
        for edge in layout.edge_list:
            if (
                (edge.u in normal_qubits and edge.v in normal_qubits) or
                (edge.u in rest_qubits and edge.v in rest_qubits)
            ):
                new_topo.add_edge(edge.u, edge.v, edge.directional)

        cir = Circuit(width, topology=new_topo)
        size = width * 10
        pro_s = 0.8  # the probability of one qubit gate in all circuit
        len_s, len_d = len(gateset.one_qubit_gates), len([gateset.two_qubit_gate])
        prob = [pro_s / len_s] * len_s + [(1 - pro_s) / len_d] * len_d
        cir.random_append(
            size,
            gateset.one_qubit_gates + [gateset.two_qubit_gate],
            probabilities=prob,
            random_params=True
        )

        depth = cir.depth()
        cir.name = "+".join(
            ["benchmark", "highly_entangled", f"w{width}_s{size}_d{depth}_level{level}"]
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
        size = width * 10
        # build random circuit
        pro_s = 0.8  # the probability of one qubit gate in all circuit
        len_s, len_d = len(gateset.one_qubit_gates), len([gateset.two_qubit_gate])
        prob = [pro_s / len_s] * len_s + [(1 - pro_s) / len_d] * len_d
        cir.random_append(size - width, gateset.one_qubit_gates + [gateset.two_qubit_gate], probabilities=prob)
        # insert measure to circuit
        for _ in range(width):
            index = random.choice(list(range(width)))
            mea_index = random.choice(list(range(int(size * 0.25), int(size * 0.75))))
            cir.insert(Measure & index, mea_index)

        depth = cir.depth()
        cir.name = "+".join(
            ["benchmark", "mediate_measure", f"w{width}_s{size}_d{depth}_level{level}"]
        )

        return cir

    @classmethod
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
