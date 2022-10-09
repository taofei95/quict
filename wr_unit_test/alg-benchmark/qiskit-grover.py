import tempfile
import time
from QuICT.algorithm.quantum_algorithm.CNF.cnf import CNFSATOracle
from QuICT.algorithm.quantum_algorithm.grover import Grover
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from unit_test.algorithm.quantum_algorithm.grover_unit_test import main_oracle

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.circuit.library.phase_oracle import PhaseOracle
from qiskit.algorithms import AmplificationProblem

f = open("qiskit_alg_benchmark_grover_data.txt", "w+")
sim = ConstantStateVectorSimulator(matrix_aggregation=False)
backend = Aer.get_backend('aer_simulator')
quantum_instance = QuantumInstance(backend, shots=1)

# for n in range(5, 6):
#     # N = 2 ** n
#     # print(N)
#     for target in range(0, 1):
#         f = [target]
#         k, oracle = main_oracle(n, f)
#         stime = time.time()
#         grover = Grover(simulator=sim)
#         result = grover.run(n, k, oracle)
#         ttime = time.time()
#         # cir = grover.circuit(n, k, oracle)
#         print(f"Quict time : {ttime - stime}\n")


input_3sat_instance = '''
c example DIMACS-CNF 5-SAT
p cnf 5 5
2 3 5 0
-2 -4 5 0
2 4 5 0
1 3 5 0
1 3 -5 0
'''
fp = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
fp.write(input_3sat_instance)
file_name = fp.name
fp.close()
oracle = PhaseOracle.from_dimacs_file(file_name)
problem = AmplificationProblem(oracle, is_good_state=oracle.evaluate_bitstring)

from qiskit.algorithms import Grover
sstime = time.time()
grover = Grover(quantum_instance=quantum_instance)
result = grover.amplify(problem)
tttime = time.time()
print(f"Qiskit time : {tttime - sstime}\n")


filename_test =  "wr_unit_test/alg-benchmark/3_5_5"
cnf = CNFSATOracle()
cnf.run(filename_test, 3)
cgate = cnf.circuit()