import os
import numpy as np

from QuICT.algorithm.quantum_algorithm import CNFSATOracle, Grover
from QuICT.simulation.state_vector import StateVectorSimulator


# 导入CNF文件，及读取相关参数
filename_test = os.path.join(os.path.dirname(__file__), "test.cnf")
variable_number, clause_number, CNF_data = CNFSATOracle.read_CNF(filename_test)
solutions = CNFSATOracle.find_solution_count(
    variable_number,
    clause_number,
    CNF_data
)

# 生成 Grover + CNF 量子线路
simulator = StateVectorSimulator()
cnf = CNFSATOracle(simulator)
oracle = cnf.circuit(
    cnf_para=[variable_number, clause_number, CNF_data],
    ancilla_qubits_num=5,
    dirty_ancilla=1
)
grover = Grover(simulator)
grover_circ = grover.circuit(
    n=variable_number,
    n_ancilla=6,
    oracle=oracle,
    n_solution=solutions,
    measure=False,
    is_bit_flip=True
)
simulator.run(grover_circ)
result_samples = simulator.sample(1000)

# 结果验证
result_var_samples = np.array(result_samples).reshape(
    (1 << variable_number, 1 << 6)
).sum(axis=1)
n_hit = 0
for result in range(1 << variable_number):
    result_str = bin(result)[2:].rjust(variable_number, '0')
    if CNFSATOracle.check_solution([int(x) for x in result_str], variable_number, clause_number, CNF_data):
        n_hit += result_var_samples[result]

print(f"success rate [{n_hit}/{1000}]:{n_hit/1000:.3f}")
