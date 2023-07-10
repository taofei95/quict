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
        typelist = [random.choice(gateset.one_qubit_gates), gateset.two_qubit_gate]
        prob = [0.8, 0.2]
        layout_list = layout.edge_list

        gate_indexes = list(range(2))
        qubits_indexes = list(range(width))
        shuffle_qindexes = qubits_indexes[:]
        random.shuffle(shuffle_qindexes)

        gate_prob = range(2 + (level - 1) * 4, 2 + level * 4)
        random_para = round(level / 3, 4)

        cirs_list = []
        for g in gate_prob:
            cir = Circuit(width)
            size = width * g
            while cir.size() < size:
                rand_type = np.random.choice(gate_indexes, p=prob)
                gate_type = typelist[rand_type]
                gate = gate_builder(gate_type)

                if gate.params:
                    gate.pargs = list(np.random.uniform(0, 2 * np.pi, gate.params))

                gsize = gate.controls + gate.targets
                if gsize > len(shuffle_qindexes):
                    continue
                if gsize == 2:
                    inset_index = np.random.choice(layout_list)
                    insert_idx = random.choice(list(range(width)))
                    cir.insert(gate & [inset_index.u, inset_index.v], insert_idx)
                else:
                    gate | cir(shuffle_qindexes[:gsize])
                if gsize == len(shuffle_qindexes) or random.random() > random_para:
                    shuffle_qindexes = qubits_indexes[:]
                    random.shuffle(shuffle_qindexes)
                else:
                    shuffle_qindexes = shuffle_qindexes[gsize:]

            depth = cir.depth()
            cir.name = "+".join(
                ["benchmark", "highly_parallelized", f"w{width}_s{size}_d{depth}_v{random_para}_level{level}"]
            )
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def serialized_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        gate_prob = range(2 + (level - 1) * 4, 2 + level * 4)
        random_para = round(level / 3, 4)

        cirs_list = []
        base_gate = gate_builder(gateset.two_qubit_gate)
        layout_list = layout.edge_list

        for g in gate_prob:
            size = width * g
            temp_size, void_gates = 0, 0

            cir = Circuit(width)
            temp_size += 1
            qubit_indexes = list(range(width))
            qubit = random.choice(qubit_indexes)
            qubit_indexes.remove(qubit)
            while temp_size < size:
                qubit_new = random.choice(qubit_indexes)
                qubits_list = [qubit, qubit_new]
                random.shuffle(qubits_list)
                base_gate | cir(qubits_list)
                temp_size += 1
                if random.random() > random_para:
                    inset_index = np.random.choice(layout_list)
                    cir.insert(base_gate & [inset_index.u, inset_index.v], qubit)
                    temp_size += 1
                    void_gates += 1

            depth = cir.depth()
            void_gates = round(void_gates / size, 2)
            cir.name = "+".join(
                ["benchmark", "highly_serialized", f"w{width}_s{size}_d{depth}_v{void_gates}_level{level}"]
            )
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def entangled_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        base_gate = gate_builder(gateset.two_qubit_gate)

        def _pattern1():
            cgate = CompositeGate()
            qubit_indexes = list(range(width))
            qubit_extra = []
            for _ in range(width):
                if len(qubit_indexes) > 1:
                    qubit_index = random.sample(qubit_indexes, 2)
                    base_gate | cgate(qubit_index)
                    qubit_extra.append(qubit_index)
                    qubit_indexes = list(set(qubit_indexes) - set(qubit_index))
                elif len(qubit_indexes) == 1:
                    for i in range(len(qubit_extra)):
                        q_collect = random.choice(qubit_extra[i])
                        base_gate | cgate([qubit_indexes[0], q_collect])
                    break
                else:
                    break
            return cgate

        def _pattern2():
            cgate = CompositeGate()
            qubit_indexes = list(range(width))
            result = [qubit_indexes[i:i + 2] for i in range(0, len(qubit_indexes), 2)]
            for i in range(len(result)):
                if len(result[i]) == 2:
                    base_gate | cgate(result[i])

            result = [qubit_indexes[i + 1:i + 3] for i in range(0, len(qubit_indexes), 2)]
            for i in range(len(result)):
                if len(result[i]) == 2:
                    base_gate | cgate(result[i])
                else:
                    break

            return cgate

        gate_prob = range(2 + (level - 1) * 4, 2 + level * 4)
        random_para = round(level / 3, 4)

        cirs_list = []
        layout_list = layout.edge_list
        for g in gate_prob:
            void_gates = 0
            cir = Circuit(width)
            size = width * g
            while cir.size() < size:
                if size - cir.size() < width or random.random() > random_para:
                    inset_index = np.random.choice(layout_list)
                    qubit_indexes = list(range(width))
                    qubit = random.choice(qubit_indexes)
                    cir.insert(base_gate & [inset_index.u, inset_index.v], qubit)
                    void_gates += 1
                else:
                    cgate = random.choice([_pattern1(), _pattern2()])
                    cgate | cir

            depth = cir.depth()
            void_gates = round(void_gates / size, 2)
            cir.name = "+".join(
                ["benchmark", "highly_entangled", f"w{width}_s{size}_d{depth}_v{void_gates}_level{level}"]
            )
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def mediate_measure_circuit_build(width: int, level: int, gateset: InstructionSet, layout: Layout):
        typelist = [random.choice(gateset.one_qubit_gates), gateset.two_qubit_gate]
        prob = [0.8, 0.2]

        layout_list = layout.edge_list

        def flat_build():
            cgate = CompositeGate()
            inset_index = np.random.choice(layout_list)
            gate_indexes = list(range(2))
            rand_type = np.random.choice(gate_indexes, p=prob)
            rand_one_qubit = int(np.random.choice(list(range(width))))
            gate_type = typelist[rand_type]
            gate = gate_builder(gate_type)
            if gate.controls + gate.targets == 1:
                cgate.insert(gate & rand_one_qubit, rand_one_qubit)
            else:
                gate & [inset_index.u, inset_index.v] | cgate

            return cgate

        cir_list = []
        gate_prob = range(2 + (level - 1) * 4, 2 + level * 4)
        random_para = round(level / 3, 4)

        for g in gate_prob:
            size = width * g
            void_gates = 0
            cir = Circuit(width)
            for _ in range(size - 2 * width, size + 2 * width):
                flat_build() | cir
            Measure | cir
            while cir.size() < size:
                if size - cir.size() < width or random.random() > random_para:
                    cir.random_append(1, [random.choice(typelist)])
                    void_gates += 1
                else:
                    flat_build() | cir

            depth = cir.depth()
            void_gates = round(void_gates / depth, 2)
            cir.name = "+".join(
                ["benchmark", "mediate_measure", f"w{width}_s{size}_d{depth}_v{random_para}_level{level}"]
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
