from telnetlib import GA
from tokenize import Double
import numpy as np
from QuICT.core.circuit import Circuit
from QuICT.core.utils.gate_type import GateType
from QuICT.core.gate import build_gate
from QuICT.qcda.mapping.mcts.mcts_mapping import MCTSMapping

single_typelist = [GateType.rz, GateType.sx, GateType.x]
double_typelist = [GateType.cx]
instruction_set = ["IBMQ"]
len_s, len_d = len(single_typelist), len(double_typelist)
prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_d

#build random circuit for qcda benchmark
qubit_num = [5, 6, 7, 8, 9, 10]
gate_multiply = [5, 7, 9, 11, 13, 15]
folder_path = "wr_unit_test/benchmark/Instruction_set/IBMQ1"
for q_num in qubit_num:
    for gm in gate_multiply:
        for i in range(1, 6):
            cir = Circuit(q_num)
            cir.random_append(q_num * gm, single_typelist+double_typelist, probabilities=prob)
            dep = cir.depth()
            file = open(folder_path + '/' + f"q{q_num}-g{gm*q_num}-IBMQ-{i}.qasm",'w+')
            file.write(cir.qasm())
            file.close()
