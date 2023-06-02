import os

from QuICT.algorithm.quantum_algorithm import CNFSATOracle
from QuICT.simulation.state_vector import StateVectorSimulator


# Read CNF files
filename_test = os.path.join(os.path.dirname(__file__), "test.cnf")
variable_number, clause_number, CNF_data = CNFSATOracle.read_CNF(filename_test)
# Find CNF Solutions
solutions = CNFSATOracle.find_solution_count(variable_number, clause_number, CNF_data)
print(solutions)

# Get CNF Circuit
cnf = CNFSATOracle(StateVectorSimulator())
circ = cnf.circuit([variable_number, clause_number, CNF_data], 3, 1, output_cgate=False)
circ.draw(method="command", flatten=True)

# Solve CNF
sim = StateVectorSimulator()
sv = sim.run(circ)
sample_result = sim.sample(100)
print(sample_result)
