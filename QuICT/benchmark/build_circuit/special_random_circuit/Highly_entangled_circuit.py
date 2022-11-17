import os
import random
import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.utils.gate_type import GateType
from QuICT.core.gate import *

# get Highly engtangled circuit
def He_circuit_build(
    qubits: int,
    rand_size: int,
    typelist: list = None,
    random_params: bool = True,
    probabilities: list = None

):
    if typelist is None:
        typelist = [GateType.cx, GateType.h]

        cir = Circuit(qubits)
        cir.random_append(rand_size=rand_size, typelist=typelist, probabilities=probabilities, random_params=random_params)

    Measure | cir

    # cir.draw(filename='entangled')

    return cir

gate_multiply = []
for i in range(5, 26):
    gate_multiply.append(i)
    
folder_path = "QuICT/lib/circuitlib/circuit_qasm/random/Highly_entangled"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for q_num in range(2, 31):
    for gm in gate_multiply:
            for i in range(1):
                cir = He_circuit_build(q_num, q_num * gm, random_params=True)
                file = open(folder_path + '/' + f"w{q_num}_s{cir.size()}_d{cir.depth()}.qasm",'w+')
                file.write(cir.qasm())
                file.close()






# nt = cir.count_2qubit_gate()

# E = nt/ng
# print(E)
# print(cir.size(), cir.depth())

# f = open("Highly_entangled_circuit.qasm", 'w+')
# f.write(cir.qasm())

  

