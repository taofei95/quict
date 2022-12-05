import random
import numpy as np

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *

# get Highly serialized circuit
def Hs_circuit_build(
    qubits: int,
    rand_size: int ,
):
    cir = Circuit(qubits)
    qubit_indexes = list(range(qubits))
    qubits_indexes = ['control_qubit', 'target_qubit']

    qubit = random.choice(qubit_indexes)
    qubits_index = random.choice(qubits_indexes)
    qubits_index = qubit
    qubit_indexes.remove(qubit)
    print(qubits_index)
    
    while cir.size() < rand_size:
        qubit_new = random.choice(qubit_indexes)
        qubits_index = [x for x in qubits_indexes if x != qubits_index]
        qubits_index = qubit_new
        a_list = [qubit, qubit_new]
        index_list = [random.choice(a_list), random.choice(a_list)]
        if index_list[0] != index_list[1]:
            CX & (index_list)| cir
    print(cir.draw(filename="hs"))
    

Hs_circuit_build(5, 10)


# gate_multiply = []
# for i in range(5, 26):
#     gate_multiply.append(i)
    
# # folder_path = "QuICT/lib/circuitlib/circuit_qasm/random/Highly_serialized"
# # if not os.path.exists(folder_path):
# #     os.makedirs(folder_path)

# for q_num in range(2, 31):
#     for gm in gate_multiply:
#             for i in range(1):
#                 cir = Hs_circuit_build(q_num, q_num * gm, random_params=True)
#                 file = open(folder_path + '/' + f"w{q_num}_s{cir.size()}_d{cir.depth()}.qasm",'w+')
#                 file.write(cir.qasm())
#                 file.close()



