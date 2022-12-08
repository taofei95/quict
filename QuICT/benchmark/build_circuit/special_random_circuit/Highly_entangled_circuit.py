import random

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *

# get Highly engtangled circuit
def He_circuit_build(
    qubits: int,
    rand_size: int,
):
    cir = Circuit(qubits)
    
    def filter(qubit_indexes, qubit_index):
        qubit_index_new = []
        for i in qubit_indexes:
            if i not in qubit_index:
                qubit_index_new.append(i)
        return qubit_index_new
     
    # a = [0,3,4,5,6]
    # b = [6,4]
    # print(delete(a,b))  [0,3,5]
    
    def delete(qubit_indexes, qubit_index):
        qubit_stayed = []
        for i_1 in qubit_index:
            for j_1 in qubit_index:
                if i_1 != j_1:
                    i = min(i_1, j_1)
                    j = max(i_1, j_1)
        qubit_extra = []            
        if abs(i-j) > 1:
            qubit_extra = qubit_indexes[qubit_indexes.index(i)+1:qubit_indexes.index(j)]
            for m in qubit_indexes:
                if m<i or m>j:
                    qubit_stayed.append(m)
        if abs(i-j) == 1:
            for m in qubit_indexes:
                    if m<i or m>j:
                        qubit_stayed.append(m)
            
        return qubit_stayed, qubit_extra
    
    # a = [0,3,4,5,6]
    # b = [6,4]
    # print(delete(a,b))   [0,3]
    
    # def build1():
    #     qubit_indexes = list(range(qubits))  #[0, 1, 2, 3, 4]
    #     if len(qubit_indexes) > 1:
    #         qubit_index = random.sample(qubit_indexes, 2)   #[a, b]
    #         CX & (qubit_index)| cir    
    #         qubit_indexes = filter(qubit_indexes, qubit_index)  #[0, 1, 2, 3, 4] - [a, b]
    #     elif len(qubit_indexes) == 1:
    #         H & (qubit_indexes)| cir
    #     return cir
    
    # while cir.size() < rand_size:
    #     cir = build1()
    #     print(cir.qasm())
    
   
    
    def build2():
        qubit_indexes = list(range(qubits))
        for i in range(qubits):
            if len(qubit_indexes) > 1:
                qubit_index = random.sample((qubit_indexes), 2)
                CX & (qubit_index)| cir   
                qubit_indexes, qubit_extra = delete(qubit_indexes, qubit_index)
                for j in qubit_index:
                    for a in qubit_indexes:
                        if abs(j-a)==1:
                            CX & ([j, a])| cir  
                            qubit_indexes.remove(a)      
            elif len(qubit_indexes) == 1:
                H & (qubit_indexes) | cir
                break
            else:
                break
        
        for i in range(qubits):
            if len(qubit_extra) != 0: 
                for i in range(len(qubit_extra)):
                    if len(qubit_extra) > 1:
                        qubit_i = random.sample((qubit_extra), 2)
                        CX & (qubit_i)| cir   
                        qubit_extra = filter(qubit_extra, qubit_i)
                    elif len(qubit_extra) == 1:
                        H & (qubit_extra)| cir 
                        qubit_extra = filter(qubit_extra, qubit_extra)
            else:
                break
            
        return cir
                        
    print(build2().qasm())
    
    while cir.size() < rand_size:
        cir = build2()
        print(cir.qasm())
 

    cir.draw(filename="he")

He_circuit_build(5, 5)


































# def Hp_circuit_build(
#     qubits: int,
#     rand_size: int,
#     typelist: list = None,
#     random_params: bool = True,
#     probabilities: list = None
# ):
#     if typelist is None:
#         single_typelist = [GateType.rz]
#         double_typelist = [GateType.cx]
#         typelist = single_typelist + double_typelist
#         len_s, len_d = len(single_typelist), len(double_typelist)
#         prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_s
        
#     gate_indexes = list(range(len(typelist)))
#     qubits_indexes = list(range(qubits))
#     shuffle_qindexes = qubits_indexes[:]
#     random.shuffle(shuffle_qindexes)

#     cir = Circuit(qubits)
#     while cir.size() < rand_size:
#         rand_type = np.random.choice(gate_indexes, p=prob)
#         gate_type = typelist[rand_type]
#         gate = GATE_TYPE_TO_CLASS[gate_type]()

#         if random_params and gate.params:
#             gate.pargs = list(np.random.uniform(0, 2 * np.pi, gate.params))

#         gsize = gate.controls + gate.targets
#         if gsize > len(shuffle_qindexes):
#             continue

#         gate & shuffle_qindexes[:gsize] | cir

#         if gsize == len(shuffle_qindexes):
#             shuffle_qindexes = qubits_indexes[:]
#             random.shuffle(shuffle_qindexes)
#         else:
#             shuffle_qindexes = shuffle_qindexes[gsize:]

#     Measure | cir

#     # cir.draw(filename='parallelized')

#     return cir
    
# def Hs_circuit_build(
#     qubits: int,
#     rand_size: int ,
#     typelist: list = None,
#     random_params: bool = True,
#     probabilities: list = None
# ):
#     if typelist is None:
#         typelist = [
#             GateType.cx
#         ]

#     gate_indexes = list(range(len(typelist)))
#     qubits_indexes = list(range(qubits))
#     shuffle_qindexes = qubits_indexes[:]
#     random.shuffle(shuffle_qindexes)

#     cir = Circuit(qubits)
#     while cir.size() < rand_size:
#         rand_type = np.random.choice(gate_indexes, p=probabilities)
#         gate_type = typelist[rand_type]
#         gate = GATE_TYPE_TO_CLASS[gate_type]()

#         if random_params and gate.params:
#             gate.pargs = list(np.random.uniform(0, 2 * np.pi, gate.params))

#         gsize = gate.controls + gate.targets
#         if gsize > len(shuffle_qindexes):
#             continue

#         gate & shuffle_qindexes[:gsize] | cir


#     Measure | cir

#     # cir.draw(filename='Highly_serialized')

#     return cir

# def Mm_circuit_build(
#     qubits: int,
#     rand_size: int,
#     typelist: list = None,
#     random_params: bool = False,
# ):
#     if typelist is None:
#         single_typelist = [GateType.rz]
#         double_typelist = [GateType.cx]
#         typelist = single_typelist + double_typelist
#         len_s, len_d = len(single_typelist), len(double_typelist)
#         prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_s

#     gate_indexes = list(range(len(typelist)))
#     qubits_indexes = list(range(qubits))
#     shuffle_qindexes = qubits_indexes[:]
#     random.shuffle(shuffle_qindexes)

#     cir = Circuit(qubits)        
#     while cir.size() < rand_size:
#         rand_type = np.random.choice(gate_indexes, p=prob)
#         gate_type = typelist[rand_type]
#         gate = GATE_TYPE_TO_CLASS[gate_type]()

#         if random_params and gate.params:
#             gate.pargs = list(np.random.uniform(0, 2 * np.pi, gate.params))

#         gsize = gate.controls + gate.targets
#         if gsize > len(shuffle_qindexes):
#             continue

#         gate & shuffle_qindexes[:gsize] | cir

#         if gsize == len(shuffle_qindexes):
#             shuffle_qindexes = qubits_indexes[:]
#             random.shuffle(shuffle_qindexes)
#         else:
#             shuffle_qindexes = shuffle_qindexes[gsize:]

#         if cir.size() == rand_size/2:
#             Measure | cir
#             continue

#     # cir.draw(filename='mediate measure')

#     return cir

# folder_path = "QuICT/benchmark/test"
# if not os.path.exists(folder_path):
#     os.makedirs(folder_path)

# for q_num in range(2, 16):
#     for q_size in range(6, 16):
#                 cir = He_circuit_build(q_num, q_size, random_params=True)
#                 cir_1 = Hp_circuit_build(q_num, q_size, random_params=True)
#                 cir_2 = Hs_circuit_build(q_num, q_size, random_params=True)
#                 cir_3 = Mm_circuit_build(q_num, q_size, random_params=True)
                
#                 file = open(folder_path + '/' + f"He_w{q_num}_s{cir.size()}_d{cir.depth()}.qasm",'w+')
#                 file = open(folder_path + '/' + f"Hp_w{q_num}_s{cir_1.size()}_d{cir_1.depth()}.qasm",'w+')
#                 file = open(folder_path + '/' + f"Hs_w{q_num}_s{cir_2.size()}_d{cir_2.depth()}.qasm",'w+')
#                 file = open(folder_path + '/' + f"Mm_w{q_num}_s{cir_3.size()}_d{cir_3.depth()}.qasm",'w+')

#                 file.write(cir.qasm())
#                 file.close()
# ltime = time.time()
# print(ltime - stime)