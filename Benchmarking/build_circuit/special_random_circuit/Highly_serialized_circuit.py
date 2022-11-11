import random
import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *

# get Highly serialized circuit
def Hs_circuit_build(
    qubits: int,
    rand_size: int = 10,
    typelist: list = None,
    random_params: bool = False,
    probabilities: list = None
):
    if typelist is None:
        typelist = [
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


    Measure | cir

    return cir

cir = circuit_build(5, 20)
cir.draw(filename='serialized')

ng = cir.size()
d = cir.depth()
print(ng, d)

f = open("Highly_serialzed_circuit.qasm", 'w+')
f.write(cir.qasm())

