import random
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *


class BenchmarkCircuitBuilder:
    """
    A class fetch QuICT benchmark circuits.

     Args:
        max_width(int): max number of qubits
        max_size(int): max number of gates
    """
    @staticmethod
    def Hp_circuit_build(width: int, size: int, random_params: bool = True):
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

    def Hs_circuit_build(self):
        cir = Circuit(self._max_width)
        qubit_indexes = list(range(self._max_width))
        qubits_indexes = ['control_qubit', 'target_qubit']

        qubit = random.choice(qubit_indexes)
        qubits_index = random.choice(qubits_indexes)
        qubits_index = qubit
        qubit_indexes.remove(qubit)

        while cir.size() < self._max_size:
            qubit_new = random.choice(qubit_indexes)
            qubits_index = [x for x in qubits_indexes if x != qubits_index]
            qubits_index = qubit_new
            qubits_list = [qubit, qubit_new]
            index_list = [random.choice(qubits_list), random.choice(qubits_list)]
            if index_list[0] != index_list[1]:
                CX & (index_list) | cir

        return cir

    def He_circuit_build(self):
        cir = Circuit(self._max_width)

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
            while cir.size() < self._max_size:
                qubit_indexes = list(range(self._max_width))
                if len(qubit_indexes) > 1:
                    qubit_index = random.sample(qubit_indexes, 2)
                    CX & (qubit_index) | cir
                    qubit_indexes = filter(qubit_indexes, qubit_index)
                elif len(qubit_indexes) == 1:
                    for q_single in qubit_indexes:
                        for q_collect in list(range(self._max_width)):
                            CX & (q_single, q_collect) | cir
            return cir

        def build_circuit_function2():
            while cir.size() < self._max_size:
                qubit_indexes = list(range(self._max_width))
                for i in range(self._max_width):
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
                            q_collect = random.choice([x for x in list(range(self._max_width)) if x != q_single])
                            CX & ([q_single, q_collect]) | cir
                            break
                    else:
                        break

                for i in range(self._max_width):
                    if len(qubit_extra) != 0:
                        for i in range(len(qubit_extra)):
                            if len(qubit_extra) > 1:
                                qubit_i = random.sample((qubit_extra), 2)
                                CX & (qubit_i) | cir
                                qubit_extra = filter(qubit_extra, qubit_i)
                            elif len(qubit_extra) == 1:
                                for q_single in qubit_extra:
                                    q_col = random.choice([x for x in list(range(self._max_width)) if x != q_single])
                                    CX & ([q_single, q_col]) | cir
                                qubit_extra = filter(qubit_extra, qubit_extra)
                    else:
                        break
            return cir

        choice_list = [build_circuit_function1, build_circuit_function2]
        cir = random.choice(choice_list)()

        return cir

    def Mm_circuit_build(self):
        single_typelist = [GateType.rz]
        double_typelist = [GateType.cx]
        typelist = single_typelist + double_typelist
        len_s, len_d = len(single_typelist), len(double_typelist)
        prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_s

        gate_indexes = list(range(len(typelist)))
        qubits_indexes = list(range(self._max_width))
        shuffle_qindexes = qubits_indexes[:]
        random.shuffle(shuffle_qindexes)

        cir = Circuit(self._max_width)
        while cir.size() < self._max_size:
            rand_type = np.random.choice(gate_indexes, p=prob)
            gate_type = typelist[rand_type]
            gate = GATE_TYPE_TO_CLASS[gate_type]()

            if self._random_params and gate.params:
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

            if cir.size() == self._max_size / 2:
                Measure | cir
                continue

        return cir
