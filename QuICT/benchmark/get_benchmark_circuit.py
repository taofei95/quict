import random
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *


class BenchmarkCircuitBuilder:
    """
    A class fetch QuICT benchmark circuits.

     Args:
        width(int): number of qubits
        size(int): number of gates
        random_params (bool, optional): whether random parameter or use default parameter. Defaults to True.
    """
    @staticmethod
    def parallelized_circuit_build(width: int, size: int, random_params: bool = True):
        typelist = [GateType.rz, GateType.cx]
        prob = [0.8, 0.2]

        gate_indexes = list(range(2))
        qubits_indexes = list(range(width))
        shuffle_qindexes = qubits_indexes[:]
        random.shuffle(shuffle_qindexes)

        cir = Circuit(width)
        while cir.size() < size:
            rand_type = np.random.choice(gate_indexes, p=prob)
            gate_type = typelist[rand_type]
            gate = GATE_TYPE_TO_CLASS[gate_type]()

            if random_params and gate.params:
                gate.pargs = list(np.random.uniform(0, 2 * np.pi, gate.params))

            gsize = gate.controls + gate.targets
            if gsize > len(shuffle_qindexes):
                continue

            gate & shuffle_qindexes[:gsize] | cir

            if gsize == len(shuffle_qindexes):
                shuffle_qindexes = qubits_indexes[:]
                random.shuffle(shuffle_qindexes)
            else:
                shuffle_qindexes = shuffle_qindexes[gsize:]

        return cir

    @staticmethod
    def serialized_circuit_build(width: int, size: int, random_params: bool = True):
        cir = Circuit(width)
        qubit_indexes = list(range(width))
        qubits_indexes = ['control_qubit', 'target_qubit']

        qubit = random.choice(qubit_indexes)
        qubits_index = random.choice(qubits_indexes)
        qubits_index = qubit
        qubit_indexes.remove(qubit)

        while cir.size() < size:
            qubit_new = random.choice(qubit_indexes)
            qubits_index = [x for x in qubits_indexes if x != qubits_index]
            qubits_index = qubit_new
            qubits_list = [qubit, qubit_new]
            index_list = [random.choice(qubits_list), random.choice(qubits_list)]
            if index_list[0] != index_list[1]:
                CX & (index_list) | cir

        return cir

    @staticmethod
    def entangled_circuit_build(width: int, size: int, random_params: bool = True):
        cir = Circuit(width)

        def filter(qubit_indexes, qubit_index):
            qubit_index_new = []
            for i in qubit_indexes:
                if i not in qubit_index:
                    qubit_index_new.append(i)
            return qubit_index_new

        def delete(qubit_indexes, qubit_index):
            qubit_stayed = []
            for i_1 in qubit_index:
                for j_1 in qubit_index:
                    if i_1 != j_1:
                        i = min(i_1, j_1)
                        j = max(i_1, j_1)
            qubit_extra = []
            if abs(i - j) > 1:
                qubit_extra = qubit_indexes[qubit_indexes.index(i) + 1:qubit_indexes.index(j)]
                for m in qubit_indexes:
                    if m < i or m > j:
                        qubit_stayed.append(m)
            if abs(i - j) == 1:
                for m in qubit_indexes:
                    if m < i or m > j:
                        qubit_stayed.append(m)

            return qubit_stayed, qubit_extra

        def build_circuit_function1():
            while cir.size() < size:
                qubit_indexes = list(range(width))
                if len(qubit_indexes) > 1:
                    qubit_index = random.sample(qubit_indexes, 2)
                    CX & (qubit_index) | cir
                    qubit_indexes = filter(qubit_indexes, qubit_index)
                elif len(qubit_indexes) == 1:
                    for q_single in qubit_indexes:
                        for q_collect in list(range(width)):
                            CX & (q_single, q_collect) | cir
            return cir

        def build_circuit_function2():
            while cir.size() < size:
                qubit_indexes = list(range(width))
                for i in range(width):
                    if len(qubit_indexes) > 1:
                        qubit_index = random.sample((qubit_indexes), 2)
                        CX & (qubit_index) | cir
                        qubit_indexes, qubit_extra = delete(qubit_indexes, qubit_index)
                        for j in qubit_index:
                            for a in qubit_indexes:
                                if abs(j - a) == 1:
                                    CX & ([j, a]) | cir
                                    qubit_indexes.remove(a)
                    elif len(qubit_indexes) == 1:
                        for q_single in qubit_indexes:
                            q_collect = random.choice([x for x in list(range(width)) if x != q_single])
                            CX & ([q_single, q_collect]) | cir
                            break
                    else:
                        break

                for x in range(width):
                    if len(qubit_extra) != 0:
                        for x in range(len(qubit_extra)):
                            if len(qubit_extra) > 1:
                                qubit_i = random.sample((qubit_extra), 2)
                                CX & (qubit_i) | cir
                                qubit_extra = filter(qubit_extra, qubit_i)
                            elif len(qubit_extra) == 1:
                                for q_single in qubit_extra:
                                    q_col = random.choice([y for y in list(range(width)) if y != q_single])
                                    CX & ([q_single, q_col]) | cir
                                qubit_extra = filter(qubit_extra, qubit_extra)
                    else:
                        break
            return cir

        choice_list = [build_circuit_function1, build_circuit_function2]
        cir = random.choice(choice_list)()

        return cir

    @staticmethod
    def mediate_measure_circuit_build(width: int, size: int, random_params: bool = True):
        typelist = [GateType.rz, GateType.cx]
        prob = [0.8, 0.2]

        cir = Circuit(width)
        cir.random_append(size, typelist, random_params, prob)
        for _ in range(width):
            idx = random.randint(width, size)
            cir.replace_gate(idx, Measure)

        return cir
