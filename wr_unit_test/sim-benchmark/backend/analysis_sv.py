import imp
import numpy as np
import re
import pandas as pd

data = []
file = 'wr_unit_test/sim-benchmark/data/qiskit_density_matrix_speed.txt'
with open(file, 'r+') as of:
    txt = of.readlines()
    for t in txt:
        data.append((re.findall(r'\d+(?:\.\d+)?', t)))
        # print(data)

df = pd.DataFrame(data)
df.to_excel("dm.xlsx",index=False)
        
data = []
file = 'wr_unit_test/sim-benchmark/data/qiskit_state_vector_speed.txt'
with open(file, 'r+') as of:
    txt = of.readlines()
    for t in txt:
        data.append((re.findall(r'\d+(?:\.\d+)?', t)))
        # print(data)

df = pd.DataFrame(data)
df.to_excel("sv.xlsx",index=False)

