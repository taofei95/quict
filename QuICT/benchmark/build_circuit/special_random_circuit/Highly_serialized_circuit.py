import random
import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *

# get Highly serialized circuit
def Hs_circuit_build(
    qubits: int,
    rand_size: int ,
    typelist: list = None,
    random_params: bool = True,
    probabilities: list = None
):
    if typelist is None:
        typelist = [
            GateType.cx
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

    # cir.draw(filename='Highly_serialized')

    return cir

gate_multiply = []
for i in range(5, 26):
    gate_multiply.append(i)
    
folder_path = "QuICT/lib/circuitlib/circuit_qasm/random/Highly_serialized"
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)

for q_num in range(2, 31):
    for gm in gate_multiply:
            for i in range(1):
                cir = Hs_circuit_build(q_num, q_num * gm, random_params=True)
                file = open(folder_path + '/' + f"w{q_num}_s{cir.size()}_d{cir.depth()}.qasm",'w+')
                file.write(cir.qasm())
                file.close()



