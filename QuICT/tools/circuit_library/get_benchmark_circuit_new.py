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
        """
        gate_prob = range(2 + (level - 1) * 4, 2 + level * 4) # the size of cir according cirs group level
        random_para = round(1 - 1 / (3 * level), 4) # the degree of control circuit parallelism
        two_qubits_size = int(width * random_para) # the number of two qubits gate that match the topology

        cirs_list = []
        for g in gate_prob:
            cir = Circuit(width)
            size = width * g
            while cir.size() < size:
                layout_list = layout.edge_list
                for _ in range(two_qubits_size):
                    qubits_indexes = list(range(width))
                    # number of double-bit gates in a layer obey width and level
                    inset_index = np.random.choice(layout_list)
                    index = [inset_index.u, inset_index.v] # the list of layout nodes
                    insert_index = random.choice(list(range(width)))
                    gate_2q = gate_builder(gateset.two_qubit_gate, random_params=True)
                    cir.insert(gate_2q & index, insert_index)
                    qubits_indexes = list(set(qubits_indexes) - set(index)) # extra qubits after insert two qubits gate
                for _ in range(int(len(qubits_indexes) * random_para)):
                    # number of one-bit gates in a layer obey level
                    one_qubit_gate = random.choice(gateset.one_qubit_gates)
                    one_qubit_index = random.choice(qubits_indexes)
                    gate_1q = gate_builder(one_qubit_gate, random_params=True)
                    gate_1q & one_qubit_index | cir

            depth = cir.depth()
            cir.name = "+".join(
                ["benchmark", "highly_parallelized", f"w{width}_s{size}_d{depth}_v{random_para}_level{level}"]
            )
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def serialized_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        """
        The number of two quantum bits interacting on the longest path of the circuit depth is close to the total number of doublets.
        """
        cirs_list, l_list, r_list = [], [], []
        layout_list = layout.edge_list
        for i in layout_list:
            if i.u not in l_list or i.v not in l_list:
                l_list.append(i.u)
                l_list.append(i.v)
        a = max(l_list, key=l_list.count) # Find the bits that appear most frequently in the topology
        for j in range(len(layout_list)):
            if layout_list[j].u == a or layout_list[j].v == a:
                r_list.append(layout_list[j]) # Associated topology nodes for identified qubits
        l_list = list(set(layout_list) - set(r_list)) # Associative topological nodes of uncertain qubits

        gate_prob = range(2 + (level - 1) * 4, 2 + level * 4) # the size of cir according cirs group level
        random_para = round(1 - 1 / (3 * level), 4) # the degree of control circuit parallelism
        gate_2q = gate_builder(gateset.two_qubit_gate, random_params=True)

        for g in gate_prob:
            size = width * g
            temp_size = 0

            cir = Circuit(width)
            while temp_size < size:
                r_index = np.random.choice(r_list)
                index = [r_index.u, r_index.v]
                insert_index = random.choice(list(range(width)))
                cir.insert(gate_2q & index, insert_index)
                temp_size += 1
                if random.random() > random_para: # Method of selecting the insertion gate
                    def insert_two_qubit():
                        l_index = np.random.choice(l_list)
                        cir.insert(gate_2q & [l_index.u, l_index.v], insert_index)
                    def insert_one_qubit():
                        one_qubit_gate = random.choice(gateset.one_qubit_gates)
                        one_qubit_index = random.choice(list(range(width)))
                        gate_1q = gate_builder(one_qubit_gate, random_params=True)
                        gate_1q & one_qubit_index | cir
                    random.choice([insert_two_qubit(), insert_one_qubit()])
                    temp_size += 1

            depth = cir.depth()
            cir.name = "+".join(
                ["benchmark", "highly_serialized", f"w{width}_s{size}_d{depth}_v{random_para}_level{level}"]
            )
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def entangled_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        """ 
        By calculating the two quantum bits interacting to a maximum value
        """
        gate_prob = range(2 + (level - 1) * 4, 2 + level * 4) # the size of cir according cirs group level
        random_para = round(1 - 1 / (3 * level), 4) # the degree of control circuit parallelism
        two_qubits_size = int(width * random_para) # the number of two qubits gate that match the topology

        cirs_list, extra_layout_list = [], []
        for g in gate_prob:
            cir = Circuit(width)
            size = width * g
            while cir.size() < size:
                layout_list = layout.edge_list
                qubits_indexes = list(range(width))
                for _ in range(two_qubits_size):
                    inset_index = np.random.choice(layout_list)
                    index = [inset_index.u, inset_index.v] # the chosen two qubits gate index
                    insert_index = random.choice(list(range(width)))
                    gate_2q = gate_builder(gateset.two_qubit_gate, random_params=True)
                    cir.insert(gate_2q & index, insert_index)
                    extra_layout_list = list(set(layout_list) - set([inset_index])) # extra layout
                    qubits_indexes = list(set(qubits_indexes) - set(index)) # extra qubits after insert two qubits gate
                    if random.random() > random_para:
                        # There is a certain probability of inserting a list of topologies that do not require the qubit
                        prob_inset_index = np.random.choice(extra_layout_list)
                        index = [prob_inset_index.u, prob_inset_index.v]
                        insert_index = random.choice(list(range(width)))
                        gate_2q = gate_builder(gateset.two_qubit_gate, random_params=True)
                        cir.insert(gate_2q & index, insert_index)
                        # Stop the loop when each qubit is highly entangled
                    if len(qubits_indexes) == 2:
                        insert_index = random.choice(list(range(width)))
                        cir.insert(gate_2q & qubits_indexes, insert_index)
                    if len(qubits_indexes) == 1:
                        one_qubit_gate = random.choice(gateset.one_qubit_gates)
                        gate_1q = gate_builder(one_qubit_gate, random_params=True)
                        gate_1q & qubits_indexes[:] | cir

            depth = cir.depth()
            cir.name = "+".join(
                ["benchmark", "highly_entangled", f"w{width}_s{size}_d{depth}_v{random_para}_level{level}"]
            )
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def mediate_measure_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        """ 
        For circuits consisting of multiple consecutive layers of gate operations, the measurement gates extract information at different
        layers for the duration of and after the execution of the programme.
        """
        cir_list = []
        gate_prob = range(2 + (level - 1) * 4, 2 + level * 4) # the size of cir according cirs group level
        random_para = round(1 - 1 / (3 * level), 4) # the degree of control circuit parallelism

        for g in gate_prob:
            size = width * g
            cir = Circuit(width)
            layout_list = layout.edge_list
            temp_size = 0
            while cir.size() < size:
                gate = gate_builder(gateset.two_qubit_gate, random_params=True)
                inset_index = np.random.choice(layout_list)
                insert_idx = random.choice(list(range(width)))
                cir.insert(gate & [inset_index.u, inset_index.v], insert_idx)
                temp_size += 1
                if temp_size > int(size / 4) and temp_size < int(size * 3 / 4): # insert measure gate between 1/4 and 3/4 of size
                    mea_index = random.choice(list(range(width)))
                    cir.insert(Measure & mea_index, mea_index)
                if random.random() > random_para:
                    one_qubit_gate = random.choice(gateset.one_qubit_gates)
                    one_qubit_index = random.choice(list(range(width)))
                    gate_1q = gate_builder(one_qubit_gate, random_params=True)
                    gate_1q & one_qubit_index | cir

            depth = cir.depth()
            cir.name = "+".join(
                ["benchmark", "mediate_measure", f"w{width}_s{size}_d{depth}_v{random_para}_level{level}"]
            )
            cir_list.append(cir)

        return cir_list

    def get_benchmark_circuit(self, qubits_interval: int, level: int, gateset: InstructionSet, layout: Layout):
        cirs_list = []
        cirs_list.extend(self.parallelized_circuit_build(qubits_interval, level, gateset, layout))
        # cirs_list.extend(self.entangled_circuit_build(qubits_interval, level, gateset, layout))
        # cirs_list.extend(self.serialized_circuit_build(qubits_interval, level, gateset, layout))
        # cirs_list.extend(self.mediate_measure_circuit_build(qubits_interval, level, gateset, layout))

        return cirs_list
