import random
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.virtual_machine.instruction_set import InstructionSet


class BenchmarkCircuitBuilder:
    """ A class fetch QuICT benchmark circuits. """

    @staticmethod
    def parallelized_circuit_build(width: int, level: int, gateset):
        """Highly parallelised circuit construction.

        Args:
            width(int): number of qubits.
            level(int): level of circuit for benchmark.
            gateset(InstructionSet): the instruction set of quantum machine.

        Returns:
            (List[Circuit]): Return the list of circuit.
        """
        typelist = [random.choice(gateset.one_qubit_gates), gateset.two_qubit_gate]
        prob = [0.8, 0.2]

        gate_indexes = list(range(2))
        qubits_indexes = list(range(width))
        shuffle_qindexes = qubits_indexes[:]
        random.shuffle(shuffle_qindexes)

        if level == 1:
            gate_prob, random_para = range(2, 6), 0.3
        elif level == 2:
            gate_prob, random_para = range(6, 10), 0.6
        elif level == 3:
            gate_prob, random_para = range(10, 14), 0.9

        cirs_list = []
        for g in gate_prob:
            cir = Circuit(width)
            H | cir(0)
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

                gate | cir(shuffle_qindexes[:gsize])
                if gsize == len(shuffle_qindexes) or random.random() > random_para:
                    shuffle_qindexes = qubits_indexes[:]
                    random.shuffle(shuffle_qindexes)
                else:
                    shuffle_qindexes = shuffle_qindexes[gsize:]

            depth = cir.depth()
            random_para = round((size / depth - 1) / (width - 1), 2)
            cir.name = "+".join([
                "benchmark", "highly_parallelized", f"w{width}_s{size}_d{depth}_v{random_para}_level{level}"
            ])
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def serialized_circuit_build(width: int, level: int, gateset):
        """Highly serialized circuit construction.

        Args:
            width(int): number of qubits.
            level(int): level of circuit for benchmark.
            gateset(InstructionSet): the instruction set of quantum machine.

        Returns:
            (List[Circuit]): Return the list of circuit.
        """
        if level == 1:
            gate_prob, random_para = range(2, 6), 0.3
        elif level == 2:
            gate_prob, random_para = range(6, 10), 0.6
        elif level == 3:
            gate_prob, random_para = range(10, 14), 0.9

        cirs_list = []
        base_gate = gate_builder(gateset.two_qubit_gate)

        for g in gate_prob:
            size = width * g
            temp_size, void_gates = 0, 0

            cir = Circuit(width)
            H | cir(0)
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
                    base_gate | cir(random.sample(list(range(width)), 2))
                    temp_size += 1
                    void_gates += 1

            depth = cir.depth()
            void_gates = round((1 - void_gates / size), 2)
            cir.name = "+".join([
                "benchmark", "highly_serialized", f"w{width}_s{size}_d{depth}_v{void_gates}_level{level}"
            ])
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def entangled_circuit_build(width: int, level: int, gateset):
        """Highly entangled circuit construction.

        Args:
            width(int): number of qubits.
            level(int): level of circuit for benchmark.
            gateset(InstructionSet): the instruction set of quantum machine.

        Returns:
            (List[Circuit]): Return the list of circuit.
        """
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

        if level == 1:
            gate_prob, random_para = range(2, 6), 0.3
        elif level == 2:
            gate_prob, random_para = range(6, 10), 0.6
        elif level == 3:
            gate_prob, random_para = range(10, 14), 0.9

        cirs_list = []
        for g in gate_prob:
            void_gates = 0
            cir = Circuit(width)
            H | cir(0)
            size = width * g
            while cir.size() < size:
                if size - cir.size() < width or random.random() > random_para:
                    cir.random_append(1, [gateset.two_qubit_gate])
                    void_gates += 1
                else:
                    cgate = random.choice([_pattern1(), _pattern2()])
                    cgate | cir

            depth = cir.depth()
            void_gates = round((1 - void_gates / size), 2)
            cir.name = "+".join([
                "benchmark", "highly_entangled", f"w{width}_s{size}_d{depth}_v{void_gates}_level{level}"
            ])
            cirs_list.append(cir)

        return cirs_list

    @staticmethod
    def mediate_measure_circuit_build(width: int, level: int, gateset):
        """Intermediate state measurement circuit construction.

        Args:
            width(int): number of qubits.
            level(int): level of circuit for benchmark.
            gateset(InstructionSet): the instruction set of quantum machine.

        Returns:
            (List[Circuit]): Return the list of circuit.
        """
        typelist = [random.choice(gateset.one_qubit_gates), gateset.two_qubit_gate]
        prob = [0.8, 0.2]

        def flat_build():
            cgate = CompositeGate()
            qubits_indexes = list(range(width))
            random.shuffle(qubits_indexes)
            while len(qubits_indexes) > 0:
                if len(qubits_indexes) == 1:
                    Rz & qubits_indexes.pop() | cgate
                else:
                    gate_type = np.random.choice(typelist, p=prob)
                    gate = gate_builder(gate_type)
                    gate_size = gate.controls + gate.targets
                    gate & qubits_indexes[:gate_size] | cgate
                    qubits_indexes = qubits_indexes[gate_size:]

            return cgate

        cir_list = []
        if level == 1:
            gate_prob, random_para = range(2, 6), 0.3
        elif level == 2:
            gate_prob, random_para = range(6, 10), 0.6
        elif level == 3:
            gate_prob, random_para = range(10, 14), 0.9

        for g in gate_prob:
            size = width * g
            void_gates = 0
            cir = Circuit(width)
            for _ in range(size - width, size, width):
                flat_build() | cir
            Measure | cir
            while cir.size() < size:
                if size - cir.size() < width or random.random() > random_para:
                    cir.random_append(1, [random.choice(typelist)])
                    void_gates += 1
                else:
                    flat_build() | cir

            depth = cir.depth()
            void_gates = round((1 - void_gates / depth), 2)
            cir.name = "+".join([
                "benchmark", "mediate_measure", f"w{width}_s{size}_d{depth}_v{void_gates}_level{level}"
            ])
            cir_list.append(cir)

        return cir_list

    def get_benchmark_circuit(self, qubits_interval: int, level: int, gateset: InstructionSet):
        """get special benchmark circuit - highly_parallelized, highly_entangled, highly_serialized, mediate_measure.

        Args:
            qubits_interval(int): number of qubits.
            level(int): level of circuit for benchmark.
            gateset(InstructionSet): the instruction set of quantum machine.

        Returns:
            (List[Circuit]): Return the list of each type circuit.
        """
        cirs_list = []
        cirs_list.extend(self.parallelized_circuit_build(qubits_interval, level, gateset))
        cirs_list.extend(self.entangled_circuit_build(qubits_interval, level, gateset))
        cirs_list.extend(self.serialized_circuit_build(qubits_interval, level, gateset))
        cirs_list.extend(self.mediate_measure_circuit_build(qubits_interval, level, gateset))

        return cirs_list
