from QuICT.core.circuit import Circuit
from QuICT.core.utils.gate_type import GateType

single_typelist = [GateType.h, GateType.rx, GateType.ry, GateType.rz] 
double_typelist = [GateType.cx]
len_s, len_d = len(single_typelist), len(double_typelist)
prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_d

#build random circuit for qcda benchmark
qubit_num = [5, 10]
gate_multiply = [50, 100]
folder_path = "wr_unit_test/machine-benchmark/qasm/quafu"
for q_num in qubit_num:
    for gm in gate_multiply:
        for i in range(1, 11):
            cir = Circuit(q_num)
            cir.random_append(rand_size=gm, typelist=single_typelist + double_typelist, probabilities=prob, random_params=True)
            file = open(folder_path + '/' + f"q{q_num}_g{gm}_quafu_{i}.qasm",'w+')
            file.write(cir.qasm())
            file.close()