from QuICT.core.utils.gate_type import GateType
from QuICT.core.circuit.circuit import Circuit
from scipy.stats import unitary_group
from QuICT.qcda.synthesis.unitary_decomposition.unitary_decomposition import UnitaryDecomposition
import os
import numpy as np

from QuICT.core import Circuit
from QuICT.core.utils.gate_type import GateType

# single_typelist = [GateType.h, GateType.rx, GateType.ry, GateType.rz]
# double_typelist = [GateType.cx]
zong_typelist = [GateType.y ,GateType.rx ,GateType.ry ,GateType.swap ,GateType.u2 ,GateType.u3]
# len_s, len_d = len(single_typelist), len(double_typelist)
# prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_d

gate_multiply = []
for i in range(5, 26):
        gate_multiply.append(i)



# print(gate_multiply)
folder_path = "QuICT/lib/circuitlib/circuit_qasm/random/unitary"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

for q_num in range(2, 31):
    for gm in gate_multiply:
            for i in range(1):
                cir = Circuit(q_num)
                cir.random_append(q_num * gm, zong_typelist)
                matrix = unitary_group.rvs(2 ** q_num)
                cir,_ = UnitaryDecomposition().execute(matrix)
                file = open(folder_path + '/' + f"w{q_num}_s{cir.size()}_d{cir.depth()}.qasm",'w+')
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