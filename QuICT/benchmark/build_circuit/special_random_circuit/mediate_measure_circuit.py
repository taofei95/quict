import os
import random
import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.utils.gate_type import GateType
from QuICT.core.gate import *

# get mediate measure circuit
def Mm_circuit_build(
    qubits: int,
    rand_size: int,
    typelist: list = None,
    random_params: bool = False,
):
    if typelist is None:
        single_typelist = [GateType.rz]
        double_typelist = [GateType.cx]
        typelist = single_typelist + double_typelist
        len_s, len_d = len(single_typelist), len(double_typelist)
        prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_s

    gate_indexes = list(range(len(typelist)))
    qubits_indexes = list(range(qubits))
    shuffle_qindexes = qubits_indexes[:]
    random.shuffle(shuffle_qindexes)

    cir = Circuit(qubits)        
    while cir.size() < rand_size:
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

        if cir.size() == rand_size/2:
            Measure | cir
            continue

    cir.draw(filename='mediate measure')

    return cir

Mm_circuit_build(5, 20, random_params=True)
# gate_multiply = []
# for i in range(5, 26):
#     gate_multiply.append(i)
    
# folder_path = "QuICT/lib/circuitlib/circuit_qasm/random/mediate_measure"
# # if not os.path.exists(folder_path):
# #     os.makedirs(folder_path)

# for q_num in range(2, 31):
#     for gm in gate_multiply:
#             for i in range(1):
#                 cir = Mm_circuit_build(q_num, q_num * gm, random_params=True)
#                 file = open(folder_path + '/' + f"w{q_num}_s{cir.size()}_d{cir.depth()}.qasm",'w+')
#                 file.write(cir.qasm())
#                 file.close()

  

