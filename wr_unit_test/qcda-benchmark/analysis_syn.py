
import numpy as np
import re

file = 'wr_unit_test/qcda-benchmark/data/qiskit_optimization_benchmark_data_3.txt'
with open(file, 'r+') as of:
    txt = of.readlines()
    for t in txt:
        data = re.findall('\d+', t)
        # print(data)

qubit_list = [5, 6, 7, 8, 9, 10]
gate_list = [5, 7, 9, 11, 13, 15]

for i in qubit_list:
    for j in gate_list:
        for d in data:
            if '103' in d:
                print(i*j, (d[2] * d[4])/2)
    
            
