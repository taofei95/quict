import os
import numpy as np

from QuICT.core import Circuit
from QuICT.core.utils.gate_type import GateType


# single_typelist = [
#     GateType.h, GateType.id, GateType.phase, GateType.rx, GateType.t,
#     GateType.ry, GateType.rz, GateType.s, GateType.sdg, 
#     GateType.tdg, GateType.u1, GateType.u2, GateType.u3
# ]
# double_typelist = [
#     GateType.ch, GateType.crz, GateType.cu1, GateType.cu3, 
#     GateType.cx, GateType.cy, GateType.cz, GateType.rxx, 
#     GateType.rzz, GateType.swap 
# ]
single_typelist = [
    GateType.h, GateType.id, GateType.phase, GateType.rx, GateType.t,
    GateType.ry, GateType.rz, GateType.s, GateType.sdg, 
    GateType.tdg, GateType.u1, GateType.u2, GateType.u3
]
double_typelist = [
    GateType.cu1, GateType.cu3, 
    GateType.cx, GateType.cy, GateType.cz, GateType.rxx, 
    GateType.rzz, GateType.swap 
]
len_s, len_d = len(single_typelist), len(double_typelist)
prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_d

#build random circuit for qcda benchmark
qubit_num = [5, 10, 15, 20]
gate_multiply = [5, 7, 9, 11, 13, 15]
folder_path = "wr_unit_test/sim-benchmark/circuit/qiskitnew"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for q_num in qubit_num:
    for gm in gate_multiply:
        for i in range(10):
            cir = Circuit(q_num)
            cir.random_append(q_num * gm, single_typelist+double_typelist, probabilities=prob)
            dep = cir.depth()
            file = open(folder_path + '/' + f"q{q_num}-g{gm*q_num}-{i}.qasm",'w+')
            file.write(cir.qasm())
            file.close()
