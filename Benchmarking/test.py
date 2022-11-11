



import numpy as np
import os
from QuICT.Benchmarking.Benchmarking import Benchmarking
from QuICT.core.layout.layout import Layout
from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator

# def load_circuit(path):
#     from QuICT.tools.interface.qasm_interface import OPENQASMInterface
#     cir = OPENQASMInterface.load_file(path).circuit

#     return cir

# def run_benchmark(cir):
#     benchmark = Benchmarking()
#     # Layout_file = f"/data/layout/line5.layout"
#     cir_run = benchmark.run(circuit=cir, bench_synthesis=True, bench_optimization=True)

    # given_data_list = [0.70710678 0.         0.70710678 0.         0.         0.
    # 0.         0.         0.         0.         0.         0.
    # 0.         0.         0.         0.         0.         0.
    # 0.         0.         0.         0.         0.         0.
    # 0.         0.         0.         0.         0.         0.
    # 0.         0.        ]

    # analysis = benchmark.analysis(cir_run, given_data=given_data_list, KL_divergence=True)
    # print(analysis)

    # sim = ConstantStateVectorSimulator()
    # sim_da = np.array(abs(sim.run(cir_run).get()))
    # # print(sim_da)
    # return cir_run


# path = f"wr_unit_test/machine-benchmark/qasm/random.qasm"
# circuit = load_circuit(path)
# result = run_benchmark(circuit)
# print(result.qasm())

b = Benchmarking(cirlib="special_circuits", qubits=5, size=50, depth=23)
c = b.get_circuit(bench_mapping=True, bench_optimization=True)

print(c.qasm())