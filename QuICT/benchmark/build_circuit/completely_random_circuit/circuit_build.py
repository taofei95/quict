from QuICT.core.utils.gate_type import GateType
from QuICT.core.circuit.circuit import Circuit
from scipy.stats import unitary_group

import os
import numpy as np

from QuICT.core import Circuit
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.synthesis.unitary_decomposition.unitary_decomposition import UnitaryDecomposition

single_typelist = [GateType.rx, GateType.ry, GateType.rz, GateType.h, GateType.x]
double_typelist = [GateType.cx]
zong_typelist = [GateType.y ,GateType.rx ,GateType.ry ,GateType.swap ,GateType.u2 ,GateType.u3]
len_s, len_d = len(single_typelist), len(double_typelist)
prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_d

gate_multiply = []
for i in range(5, 11):
        gate_multiply.append(i)
for j in range(11, 50):
        if j%2==0:
                gate_multiply.append(j)
for m in range(50, 201):
        if m % 10 ==0:
                gate_multiply.append(m)


# print(gate_multiply)
folder_path = "QuICT/Benchmarking/circuitlib/random_circuits/unitary_set"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for q_num in range(2, 31):
    for gm in gate_multiply:
            for i in range(1, 6):
                cir = Circuit(q_num)
                cir.random_append(q_num * gm, zong_typelist)
                matrix = unitary_group.rvs(2 ** q_num)
                cir,_ = UnitaryDecomposition().execute(matrix)
                dep = cir.depth()
                file = open(folder_path + '/' + f"w{q_num}_s{cir.size()}_d{dep}_{i}.qasm",'w+')
                file.write(cir.qasm())
                file.close()



#google
GateType.fsim,
[GateType.sx, GateType.sy, GateType.sw, GateType.rx, GateType.ry]
#ibmq
GateType.cx,
[GateType.rz, GateType.sx, GateType.x]
#ionq
GateType.rxx,
[GateType.rx, GateType.ry, GateType.rz]
#ustc
GateType.cx,
[GateType.rx, GateType.ry, GateType.rz, GateType.h, GateType.x]
#quafu
[GateType.h, GateType.rx, GateType.ry, GateType.rz] 
[GateType.cx]
#ctrl_diag
[GateType.crz, GateType.cu1, GateType.cz]
#ctrl_unitary 
[GateType.cx ,GateType.cy ,GateType.ch ,GateType.cu3]
#diag
[GateType.t ,GateType.rz ,GateType.z ,GateType.sdg ,GateType.tdg ,GateType.u1 ,GateType.s ,GateType.id]
#single_bit
[GateType.x ,GateType.y ,GateType.z ,GateType.u1 ,GateType.u2 ,GateType.u3 ,GateType.tdg ,GateType.sdg ,GateType.h ,GateType.s ,GateType.t ,GateType.rx ,GateType.ry ,GateType.rz]
#unitary
[GateType.y ,GateType.rx ,GateType.ry ,GateType.swap ,GateType.u2 ,GateType.u3]