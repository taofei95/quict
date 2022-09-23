import imp
import numpy as np
import re
import pandas as pd

data = []
file = 'wr_unit_test/qcda-benchmark/data/qiskit_mapping_benchmark_data.txt'
with open(file, 'r+') as of:
    txt = of.readlines()
    for t in txt:
        data.append(re.findall('\d+', t))
# print(len(data))

layouts_grid = data[1:403]
layouts_line = data[404:806]
data_list = []

# for x in range(0, len(layouts_grid), 67):
#     qubits = layouts_grid[x]
#     qubits_q = layouts_grid[x+1:x+67]
#     for y in range(0, len(qubits_q), 11):
#         gates = qubits_q[y]
#         gates_g = qubits_q[y+1:y+11]
#         quict_swap_gates_num, qiskit_swap_gates_num = 0, 0
#         for z in range(0, len(gates_g), 2):
#             quict_swap_gates_num += int(gates_g[z][1]) / 5
#             qiskit_swap_gates_num += int(gates_g[z+1][1]) / 5
#             anay_data = [quict_swap_gates_num, qiskit_swap_gates_num]

#         print(f"grid, {qubits} {gates} {anay_data}")

for x in range(0, len(layouts_line), 67):
    qubits = layouts_line[x]
    qubits_q = layouts_line[x+1:x+67]
    for y in range(0, len(qubits_q), 11):
        gates = qubits_q[y]
        gates_g = qubits_q[y+1:y+11]
        quict_swap_gates_num, qiskit_swap_gates_num = 0, 0
        for z in range(0, len(gates_g), 2):
            quict_swap_gates_num += int(gates_g[z][1]) / 5
            qiskit_swap_gates_num += int(gates_g[z+1][1]) / 5
        anay_data = [quict_swap_gates_num, qiskit_swap_gates_num]
        data_list.append(anay_data)
print(data_list)
df = pd.DataFrame(data_list)
df.to_excel("map.xlsx",index=False)
        # print(f"line, {qubits} {gates} {anay_data}")
        
    

