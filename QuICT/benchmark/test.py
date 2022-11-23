
# import os
# from QuICT.benchmark import Benchmarking
# from QuICT.benchmark.benchmark import QuICTBenchmark
# from QuICT.lib.circuitlib.circuitlib import CircuitLib


# bench = QuICTBenchmark("random", "qasm")
# cir_list = bench.get_circuit(["highly_serialized"], 5, 10, 20)

# print(cir_list)

# circuit_type = "template"
# a = CircuitLib(circuit_type)
# print(a)
# cir_list = CircuitLib().get_random_circuit("ionq", 5, 20, 20)
# print(cir_list)

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
a = 1
b = 2
c = 3

print((a+b+c)/3)