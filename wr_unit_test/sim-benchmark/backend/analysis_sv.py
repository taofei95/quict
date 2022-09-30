import imp
import numpy as np
import re
import pandas as pd

data = []
file = 'wr_unit_test/sim-benchmark/data/qiskit_state_vector_speed2.txt'
with open(file, 'r+') as of:
    txt = of.readlines()
    for t in txt:
        data.append((re.findall(r'\d+(?:\.\d+)?', t)))
        # print(len(data))
data_list = []
for x in range(0, len(data), 7):
    qubit_num = data[x]
    q_data = data[x+1:x+7]
    quict_cpu_time, quict_gpu_time, qiskit_cpu_time, qiskit_gpu_time = 0, 0, 0, 0

    for y in range(0, len(q_data), 1):
        quict_cpu_time += float(q_data[y][0])/6
        quict_gpu_time += float(q_data[y][1])/6
        qiskit_cpu_time += float(q_data[y][2])/6
        qiskit_gpu_time += float(q_data[y][3])/6
    an_data = [round(quict_cpu_time, 6), round(quict_gpu_time, 6), round(qiskit_cpu_time, 6), round(qiskit_gpu_time, 6)]
    data_list.append(an_data)
    print(data_list)

df = pd.DataFrame(data_list)
df.to_excel("dm.xlsx",index=False)
        













# data = []
# file = 'wr_unit_test/sim-benchmark/data/qiskit_state_vector_speed.txt'
# with open(file, 'r+') as of:
#     txt = of.readlines()
#     for t in txt:
#         data.append((re.findall(r'\d+(?:\.\d+)?', t)))
#         # print(data)

# df = pd.DataFrame(data)
# df.to_excel("sv.xlsx",index=False)

