
import os
from QuICT.benchmark import Benchmarking
from QuICT.benchmark.benchmark import QuICTBenchmark
from QuICT.lib.circuitlib.circuitlib import CircuitLib


bench = QuICTBenchmark("circuit", "Graph")
cir_list = bench.get_circuit(["grover", "qft", "google"], 10, 20, 20)
print(cir_list)

# print(cir_list)

# circuit_type = "template"
# a = CircuitLib(circuit_type)
# print(a)
cir_list = CircuitLib().get_algorithm_circuit("grover", 5, 20, 20)
print(cir_list)

# filePath = 'QuICT/lib/circuitlib/circuit_qasm/algorithm'
# a = os.listdir(filePath)
# c = []
# for b in a:
#     if b == '.keep':
#         continue
#     c.append(b)
# print(c)

# a_list = {"a":2, "b":3}
# for a in a_list:
#     print(a)

# file_path = 'QuICT/lib/circuitlib/circuit_qasm'
# alg_file_list = os.listdir(file_path)

# print(alg_file_list)
# for value in CircuitLib().__DEFAULT_CLASSIFY:
#     print(value)

# a = QuICTBenchmark().show_result()
