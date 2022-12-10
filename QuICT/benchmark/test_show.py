import matplotlib as mpl
from QuICT.benchmark.benchmark import QuICTBenchmark
import numpy as np
import scipy.stats

bench = QuICTBenchmark("circuit", "Table")
cir_list = bench.get_circuit(["grover"], 4, 25, 12)
for cir in cir_list[0]:
    print(cir.name)
result = [np.load("grover1.npy")]
ev = bench.evaluate(circuit_list=cir_list[0], result_list=result, output_type="Table")

print(ev)









