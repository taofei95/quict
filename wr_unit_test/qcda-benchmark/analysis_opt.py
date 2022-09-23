
from csv import excel
import numpy as np
import re


data = []
file = 'wr_unit_test/qcda-benchmark/data/qiskit_optimization_benchmark_data_3.txt'
with open(file, 'r+') as of:
    txt = of.readlines()
    for t in txt:
        data.append(re.findall('\d+', t))

print(len(data))
for x in range(0, len(data), 67):
    qubits = data[x]
    cur_q_data = data[x+1:x+67]
    for y in range(0, len(cur_q_data), 11):
        gates = cur_q_data[y]
        cur_g_data = cur_q_data[y+1:y+11]
        quict_opt_gates, quict_ori_depth, quict_opt_depth = 0, 0, 0
        qiskit_opt_gates, qiskit_ori_depth, qiskit_opt_depth = 0, 0, 0
        for z in range(0, len(cur_g_data), 2):
            quict_opt_gates += int(cur_g_data[z][2]) / 5
            quict_ori_depth += int(cur_g_data[z][3]) / 5
            quict_opt_depth += int(cur_g_data[z][4]) / 5

            qiskit_opt_gates += int(cur_g_data[z+1][2]) / 5
            qiskit_ori_depth += int(cur_g_data[z+1][3]) / 5
            qiskit_opt_depth += int(cur_g_data[z+1][4]) / 5

        anay_data = [quict_opt_gates, qiskit_opt_gates, quict_ori_depth, qiskit_ori_depth, quict_opt_depth, qiskit_opt_depth]

        print(f"{qubits} {gates} {anay_data}")



    # print(cur_q_data)

# f = open("analysis opt.txt","w+")
# for i in qubit_list:
#     for j in gate_list:
#         for d in data:
#             for h in range(1, 5):
#                 f.write(f"gate num : {i*j}, pj: {sum(d[2] * d[4])/2} \n")
    
# f.close()
        