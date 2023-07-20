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
                                  ┌───────────┐ ┌─────────────┐┌────────────┐     ┌───┐      ┌────────────┐                                                               »
            q_0: |0>──────■───────┤ ry(4.551) ├─┤ rz(0.14959) ├┤ ry(1.4667) ├─────┤ h ├──────┤ rz(1.2718) ├───────────────────────────────────────────────────────────────»
                        ┌─┴──┐    └───┬───┬───┘ └────┬───┬────┘├────────────┤┌────┴───┴────┐ └────────────┘                     ┌───────────┐    ┌───┐      ┌────────────┐»
            q_1: |0>────┤ cx ├────────┤ h ├──────────┤ h ├─────┤ rz(2.0587) ├┤ ry(0.14683) ├────────────────────────────────■───┤ rz(4.019) ├────┤ h ├──────┤ rx(5.4462) ├»
                        └────┘        └───┘          └───┘     ├───────────┬┘└────┬───┬────┘┌──────────────┐┌────────────┐┌─┴──┐└───────────┘┌───┴───┴────┐ ├────────────┤»
            q_2: |0>───────────────────────────────────■───────┤ ry(5.198) ├──────┤ h ├─────┤ ry(0.083308) ├┤ ry(1.9855) ├┤ cx ├──────■──────┤ rz(3.3053) ├─┤ rz(4.8386) ├»
                    ┌────────────┐┌────────────┐     ┌─┴──┐    └───┬───┬───┘ ┌────┴───┴────┐└┬────────────┬┘└────────────┘└────┘    ┌─┴──┐   ├────────────┤┌┴────────────┤»
            q_3: |0>┤ rx(5.8446) ├┤ rz(1.3912) ├─────┤ cx ├────────┤ h ├─────┤ rz(0.98724) ├─┤ rx(1.7792) ├─────────────────────────┤ cx ├───┤ rz(2.5129) ├┤ rx(0.70219) ├»
                    └────────────┘└────────────┘     └────┘        └───┘     └─────────────┘ └────────────┘                         └────┘   └────────────┘└─────────────┘»
        """
        cirs_list = []
        prob = [0.8, 0.2]  # Probability of occurrence of single and double qubit gates
        gates = width * 10

        cir_depth_prob = range(int(gates * level / width), int(gates * level / width + 4))
        layout_list = layout.edge_list

        for d in cir_depth_prob:
            cir = Circuit(width)
            while cir.depth() < d:
                gate_indexes = list(range(2))
                qubits_indexes = list(range(width))
                shuffle_qindexes = qubits_indexes[:]
                random.shuffle(shuffle_qindexes)

                # random choice gate from gateset
                typelist = [random.choice(gateset.one_qubit_gates), gateset.two_qubit_gate]
                rand_type = np.random.choice(gate_indexes, p=prob)
                gate_type = typelist[rand_type]
                gate = gate_builder(gate_type, random_params=True)
                gsize = gate.controls + gate.targets

                if gsize > len(shuffle_qindexes):
                    continue
                if gsize == 2:
                    inset_index = np.random.choice(layout_list)
                    insert_index = random.choice(list(range(width)))
                    cir.insert(gate & [inset_index.u, inset_index.v], insert_index)
                    shuffle_qindexes = list(set(shuffle_qindexes) - set([inset_index.u, inset_index.v]))
                elif gsize == 1:
                    index = random.choice(shuffle_qindexes)
                    insert_index = random.choice(list(range(width)))
                    cir.insert(gate & index, insert_index)
                    shuffle_qindexes = list(set(shuffle_qindexes) - set([index]))
            depth = cir.depth()
            cir.name = "+".join(
                ["benchmark", "highly_parallelized", f"w{width}_s{gates}_d{depth}_level{level}"]
            )
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def serialized_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        """
        The number of two quantum bits interacting on the longest path of the circuit depth is close to the total number
            of doublets. 
        Example:
            q_0: |0>──■───────────────────────────────────────────────■───────────────────────■─────■─────■─────────────────■───
                    ┌─┴──┐                                          ┌─┴──┐                  ┌─┴──┐┌─┴──┐┌─┴──┐            ┌─┴──┐
            q_1: |0>┤ cx ├────────■─────■─────■─────■───────────■───┤ cx ├──■─────■─────■───┤ cx ├┤ cx ├┤ cx ├──■─────■───┤ cx ├
                    └────┘      ┌─┴──┐┌─┴──┐┌─┴──┐┌─┴──┐      ┌─┴──┐└────┘┌─┴──┐┌─┴──┐┌─┴──┐└────┘└────┘└────┘┌─┴──┐┌─┴──┐└────┘
            q_2: |0>──■─────■───┤ cx ├┤ cx ├┤ cx ├┤ cx ├──■───┤ cx ├──■───┤ cx ├┤ cx ├┤ cx ├──────────────────┤ cx ├┤ cx ├──────
                    ┌─┴──┐┌─┴──┐└────┘└────┘└────┘└────┘┌─┴──┐└────┘┌─┴──┐└────┘└────┘└────┘                  └────┘└────┘      
            q_3: |0>┤ cx ├┤ cx ├────────────────────────┤ cx ├──────┤ cx ├──────────────────────────────────────────────────────
                    └────┘└────┘                        └────┘      └────┘                                                                                              
        """
        cirs_list, l_list, r_list = [], [], []
        layout_list = layout.edge_list
        for i in layout_list:
            l_list.append(i.u)
            l_list.append(i.v)
        a = max(l_list, key=l_list.count)  # Find the bits that appear most frequently in the topology
        for j in range(len(layout_list)):
            if layout_list[j].u == a or layout_list[j].v == a:
                r_list.append(layout_list[j])  # Associated topology nodes for identified qubits
        l_list = list(set(layout_list) - set(r_list))  # Associative topological nodes of uncertain qubits

        split_list = list(np.linspace(0.0, 1.0, num=10))  # the degree of control circuit parallelism
        gate_2q = gate_builder(gateset.two_qubit_gate, random_params=True)

        for i in split_list[level+3:level+7]:
            size = width * 10
            temp_size = 0

            cir = Circuit(width)
            while cir.size() < size:
                r_index = np.random.choice(r_list)
                index = [r_index.u, r_index.v]
                insert_index = random.choice(list(range(width)))
                cir.insert(gate_2q & index, insert_index)
                temp_size += 1

                if temp_size < int(size * i):
                    continue
                elif size - cir.size() >= 1:
                    l_index = np.random.choice(l_list)
                    index = [l_index.u, l_index.v]
                    insert_index = random.choice(list(range(width)))
                    cir.insert(gate_2q & index, insert_index)

            depth = cir.depth()
            cir.name = "+".join(
                ["benchmark", "highly_serialized", f"w{width}_s{size}_d{depth}_level{level}"]
            )
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def entangled_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        """
        By calculating the two quantum bits interacting to a maximum value.
        Example:
                                      ┌───┐ ┌─────────────┐┌────────────┐                                                  
            q_0: |0>────────■─────■───┤ h ├─┤ ry(0.78741) ├┤ rz(1.0012) ├──■─────────────────────■─────────■─────■─────■───
                            │   ┌─┴──┐└───┘ └─────────────┘└────────────┘┌─┴──┐                  │       ┌─┴──┐┌─┴──┐┌─┴──┐
            q_1: |0>──■─────┼───┤ cx ├──■────────────────────────■───────┤ cx ├──■───────────────┼───────┤ cx ├┤ cx ├┤ cx ├
                      │   ┌─┴──┐└────┘  │                        │       └────┘  │             ┌─┴──┐    └────┘└────┘└────┘
            q_2: |0>──┼───┤ cx ├────────┼──────────■─────────────┼───────────────┼─────■───────┤ cx ├──────■───────────■───
                    ┌─┴──┐└────┘      ┌─┴──┐     ┌─┴──┐        ┌─┴──┐          ┌─┴──┐┌─┴──┐┌───┴────┴───┐┌─┴──┐┌───┐ ┌─┴──┐
            q_3: |0>┤ cx ├────────────┤ cx ├─────┤ cx ├────────┤ cx ├──────────┤ cx ├┤ cx ├┤ rz(2.7852) ├┤ cx ├┤ h ├─┤ cx ├
                    └────┘            └────┘     └────┘        └────┘          └────┘└────┘└────────────┘└────┘└───┘ └────┘
        """
        cirs_list = []

        gates_list = [8, 10, 12, 14]
        qubits_indexes = list(range(width))
        gate_2q = gate_builder(gateset.two_qubit_gate, random_params=True)
        layout_list = layout.edge_list

        for g in gates_list:
            cir = Circuit(width)
            gates = g * width
            while cir.size() < gates:
                r_list = []
                for _ in range(level):
                    for j in range(len(layout_list)):
                        a = random.choice(qubits_indexes)
                        if layout_list[j].u != a or layout_list[j].v != a:
                            r_list.append(layout_list[j])  # Associated topology nodes for identified qubits

                r_index = np.random.choice(r_list)
                index = [r_index.u, r_index.v]
                insert_index = random.choice(list(range(width)))
                cir.insert(gate_2q & index, insert_index)
                qubits_indexes = list(set(qubits_indexes) - set(index))

                if len(qubits_indexes) > 1:
                    continue
                elif len(qubits_indexes) == 1:
                    gate = random.choice(gateset.one_qubit_gates)
                    gate_1q = gate_builder(gate, random_params=True)
                    insert_index = random.choice(list(range(width)))
                    cir.insert(gate_1q & qubits_indexes[:], insert_index)
                else:
                    qubits_indexes = list(range(width))

            depth = cir.depth()
            cir.name = "+".join(
                ["benchmark", "highly_entangled", f"w{width}_s{gates}_d{depth}_level{level}"]
            )
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def mediate_measure_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        """
        For circuits consisting of multiple consecutive layers of gate operations, the measurement gates extract information at different
        layers for the duration of and after the execution of the programme.
        Example:
                          ┌─────────┐ ┌─┐  ┌─────────┐┌─────────┐ ┌─┐  ┌─────────┐   ┌───┐   ┌─────────┐                                 
            q_0: |0>──■───┤ rx(π/2) ├─┤M├──┤ rx(π/2) ├┤ rz(π/2) ├─┤M├──┤ rx(π/2) ├───┤ h ├───┤ ry(π/2) ├─────────────────────────────────
                    ┌─┴──┐└─────────┘ └─┘  └─────────┘└─────────┘┌┴─┴┐ ├─────────┤   └┬─┬┘   ├─────────┤   ┌───┐   ┌─────────┐┌─────────┐
            q_1: |0>┤ cx ├─────■───────────────────────────■─────┤ h ├─┤ rz(π/2) ├────┤M├────┤ ry(π/2) ├───┤ h ├───┤ rx(π/2) ├┤ ry(π/2) ├
                    └────┘   ┌─┴──┐                      ┌─┴──┐  └───┘ ├─────────┤┌───┴─┴───┐├─────────┤┌──┴───┴──┐└─────────┘└─────────┘
            q_2: |0>─────────┤ cx ├────■────────■────────┤ cx ├────■───┤ rx(π/2) ├┤ rx(π/2) ├┤ rz(π/2) ├┤ rx(π/2) ├──────────────────────
                    ┌───┐    └────┘  ┌─┴──┐   ┌─┴──┐     └────┘  ┌─┴──┐└───┬─┬───┘├─────────┤└──┬───┬──┘├─────────┤┌─────────┐   ┌───┐   
            q_3: |0>┤ h ├────────────┤ cx ├───┤ cx ├─────────────┤ cx ├────┤M├────┤ ry(π/2) ├───┤ h ├───┤ rx(π/2) ├┤ ry(π/2) ├───┤ h ├───
                    └───┘            └────┘   └────┘             └────┘    └─┘    └─────────┘   └───┘   └─────────┘└─────────┘   └───┘   
        """
        cir_list = []
        layout_list = layout.edge_list
        gates_list = [8, 10, 12, 14]
        pro_s = 1 - level / 10

        for g in gates_list:
            cir = Circuit(width)
            gates = g * width
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
            cir_list.append(cir)

        return cir_list

    def get_benchmark_circuit(self, qubits_interval: int, level: int, gateset: InstructionSet, layout: Layout):
        cirs_list = []
        cirs_list.extend(self.parallelized_circuit_build(qubits_interval, level, gateset, layout))
        cirs_list.extend(self.entangled_circuit_build(qubits_interval, level, gateset, layout))
        cirs_list.extend(self.serialized_circuit_build(qubits_interval, level, gateset, layout))
        cirs_list.extend(self.mediate_measure_circuit_build(qubits_interval, level, gateset, layout))

        return cirs_list
