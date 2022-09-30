
import numpy as np
import re

import pandas as pd

# file = 'wr_unit_test/qcda-benchmark/data/qiskit_optimization_benchmark_data_3.txt'
# with open(file, 'r+') as of:
#     txt = of.readlines()
#     for t in txt:
#         data = re.findall('\d+', t)
#         # print(data)

data = []
file = 'wr_unit_test/qcda-benchmark/data/qiskit_unitary_composition_synthesis_benchmark_data.txt'
with open(file, 'r+') as of:
    txt = of.readlines()
    for t in txt:
        data.append(re.findall('\d+', t))
        # print(data)

df = pd.DataFrame(data)
df.to_excel("unitary composition analysis.xlsx",index=False)
    
            
