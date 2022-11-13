import random
import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.utils.gate_type import GateType
from QuICT.core.gate import *

# get Highly engtangled circuit
def He_circuit_build(
    qubits: int,
    rand_size: int = 10,
    typelist: list = None,
    random_params: bool = False,
    probabilities: list = None
):
    if typelist is None:
        single_typelist = [GateType.h]
        double_typelist = [GateType.cx]
        len_s, len_d = len(single_typelist), len(double_typelist)
        prob = [0.1 / len_s] * len_s + [0.9 / len_d] * len_d

        cir = Circuit(qubits)
        cir.random_append(rand_size=rand_size, typelist=single_typelist + double_typelist, probabilities=prob, random_params=random_params)

    Measure | cir

    cir.draw(filename='entangled')

    return cir



ng = 20
cir = circuit_build(5, ng, random_params=True)
nt = cir.count_2qubit_gate()

E = nt/ng
print(E)
print(cir.size(), cir.depth())

f = open("Highly_entangled_circuit.qasm", 'w+')
f.write(cir.qasm())

  

