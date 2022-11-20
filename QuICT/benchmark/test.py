
import os
from QuICT.benchmark import Benchmarking
from QuICT.lib.circuitlib.circuitlib import CircuitLib

# bench = Benchmarking(type ="random", classify="ionq", width=5, size=20, depth=20)
# cir_list = bench.get_circuit()
# cir_qcda_list = bench.qcda_circuit(cir_list)
# print(cir_qcda_list)

# cir_list = CircuitLib().get_random_circuit("ionq", 5, 20, 20)
# print(cir_list)

filePath = 'QuICT/lib/circuitlib/circuit_qasm/algorithm'
a = os.listdir(filePath)
c = []
for b in a:
    if b == '.keep':
        continue
    c.append(b)
print(c)
