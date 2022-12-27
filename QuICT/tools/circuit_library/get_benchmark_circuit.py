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
        qubit = random.choice(qubit_indexes)
        qubit_indexes.remove(qubit)

        while cir.size() < size:
            qubit_new = random.choice(qubit_indexes)
            qubits_list = [qubit, qubit_new]
            CX & (qubits_list) | cir
            random.shuffle(qubits_list)

        return cir

    @staticmethod
    def entangled_circuit_build(width: int, size: int, random_params: bool = True):
        def delete(x, y):
            if x[0] < y[1]:
                x_extra = x[x.index(y[0]) + 1:x.index(y[1])]
                del x[x.index(y[0]):x.index(y[1]) + 1]
            else:
                x_extra = x[x.index(y[1]) + 1:x.index(y[0])]
                del x[x.index(y[1]):x.index(y[0]) + 1]
            return x, x_extra

        cir = Circuit(width)

        def build_circuit():
            while cir.size() < size:
                qubit_indexes = list(range(width))
                if len(qubit_indexes) > 1:
                    qubit_index = random.sample(qubit_indexes, 2)
                    CX & (qubit_index) | cir
                    qubit_indexes = list(set(qubit_indexes) - set(qubit_index))
                elif len(qubit_indexes) == 1:
                    for q_single in qubit_indexes:
                        for q_collect in list(range(width)):
                            CX & (q_single, q_collect) | cir

                while cir.size() < size - cir.size():
                    qubit_indexes = list(range(width))
                    for _ in range(width):
                        if len(qubit_indexes) > 1:
                            qubit_index = random.sample((qubit_indexes), 2)
                            CX & (qubit_index) | cir
                            qubit_indexes, qubit_extra = delete(qubit_indexes, qubit_index)
                            for i in qubit_index:
                                for j in qubit_indexes:
                                    if abs(i - j) == 1:
                                        CX & ([i, j]) | cir
                                        qubit_indexes.remove(j)
                        elif len(qubit_indexes) == 1:
                            for q_single in qubit_indexes:
                                q_collect = random.choice([x for x in list(range(width)) if x != q_single])
                                CX & ([q_single, q_collect]) | cir
                                break
                        else:
                            break

                    for _ in range(width):
                        if len(qubit_extra) != 0:
                            for _ in range(len(qubit_extra)):
                                if len(qubit_extra) > 1:
                                    qubit_i = random.sample((qubit_extra), 2)
                                    CX & (qubit_i) | cir
                                    qubit_extra = list(set(qubit_extra) - set(qubit_i))
                                elif len(qubit_extra) == 1:
                                    for q_single in qubit_extra:
                                        q_col = random.choice([y for y in list(range(width)) if y != q_single])
                                        CX & ([q_single, q_col]) | cir
                                    qubit_extra = list(set(qubit_extra) - set(qubit_extra))
                        else:
                            break
            return cir

        while cir.size() < size:
            cir = build_circuit()

        return cir

    @staticmethod
    def mediate_measure_circuit_build(width: int, size: int, random_params: bool = True):
        typelist = [GateType.rz, GateType.cx]
        prob = [0.8, 0.2]

        cir = Circuit(width)
        cir.random_append(size, typelist, random_params, prob)
        for _ in range(width):
            idx = random.randint(2 * width, size - 1)
            qidx = random.choice(list(range(width)))
            mgate = Measure & qidx
            cir.replace_gate(idx, mgate)

        return cir
