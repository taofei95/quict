import random
import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.utils.gate_type import GateType
from QuICT.core.gate import *

# get mediate measure circuit
def Mm_circuit_build(
    qubits: int,
    rand_size: int = 10,
    typelist: list = None,
    random_params: bool = False,
    probabilities: list = None
):
    if typelist is None:
        typelist = [
            GateType.rx, GateType.ry, GateType.rz,
            GateType.cx, GateType.h
        ]

    gate_indexes = list(range(len(typelist)))
    qubits_indexes = list(range(qubits))
    shuffle_qindexes = qubits_indexes[:]
    random.shuffle(shuffle_qindexes)

    cir = Circuit(qubits)        
    while cir.size() < rand_size:
        rand_type = np.random.choice(gate_indexes, p=probabilities)
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

        if cir.size() == rand_size/2:
            Measure | cir
            continue

    cir.draw(filename='mediate measure')

    return cir

cir = circuit_build(5, 50, random_params=True)

print(cir.size(), cir.depth())

f = open("Highly_mediate_measure_circuit.qasm", 'w+')
f.write(cir.qasm())

  

