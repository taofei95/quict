

import re


data = []
file = 'wr_unit_test/qcda-benchmark/data/qiskit_optimization_benchmark_data_3.txt'
with open(file, 'r+') as of:
    txt = of.readlines()
    for t in txt:
        data.append(re.findall('\d+', t))

# print(len(data))
for x in range(0, len(data), 67):
    qubits = data[x]
    qubits_q = data[x+1, x+67]
    for y in range(0, len(qubits_q), 11):
        gates = data[y]
        gates_g = data[y+1, y+11]
        quict_opt_size, quict_ori_depth, quict_opt_depth = 0, 0, 0
        qiskit_opt_size, qiskit_ori_depth, qiskit_opt_depth = 0, 0, 0
        for z in range(0, len(gates_g), 2):
            quict_opt_size += int(gates_g[z][2])/5
            quict_opt_size += int(gates_g[z][3])/5
            quict_opt_size += int(gates_g[z][4])/5

