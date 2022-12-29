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
            random.shuffle(qubits_list)
            CX & (qubits_list) | cir

        return cir

    @staticmethod
    def entangled_circuit_build(width: int, size: int, random_params: bool = True):
        def _pattern1():
            qubit_indexes = list(range(width))
            qubit_extra = []
            for _ in range(width):
                if len(qubit_indexes) > 1:
                    qubit_index = random.sample(qubit_indexes, 2)
                    CX & (qubit_index) | cir
                    qubit_extra.append(qubit_index)
                    qubit_indexes = list(set(qubit_indexes) - set(qubit_index))
                elif len(qubit_indexes) == 1:
                    for i in range(len(qubit_extra)):
                        q_collect = random.choice(qubit_extra[i])
                        CX & ([qubit_indexes[0], q_collect]) | cir
                    break
                else:
                    break
            return cir

        def _pattern2():
            qubit_indexes = list(range(width))
            result = [qubit_indexes[i:i + 2] for i in range(0, len(qubit_indexes), 2)]
            for i in range(len(result)):
                if len(result[i]) == 2:
                    CX & (result[i]) | cir
            result = [qubit_indexes[i + 1:i + 3] for i in range(0, len(qubit_indexes), 2)]
            for i in range(len(result)):
                if len(result[i]) == 2:
                    CX & (result[i]) | cir
                else:
                    break
            return cir

        cir = Circuit(width)
        cir = random.choice([_pattern1(), _pattern2()])
        while cir.size() < size:
            cir = random.choice([_pattern1(), _pattern2()])

        return cir

    @staticmethod
    def mediate_measure_circuit_build(width: int, size: int, random_params: bool = True):
        typelist = [GateType.rz, GateType.cx]
        prob = [0.8, 0.2]

        cir = Circuit(width)
        cir.random_append(size, typelist, random_params, prob)
        for _ in range(width):
            idx = random.randint(size / width, size - 1)
            qidx = random.choice(list(range(width)))
            mgate = Measure & qidx
            cir.replace_gate(idx, mgate)

        return cir
