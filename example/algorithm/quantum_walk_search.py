from QuICT.algorithm.quantum_algorithm import QuantumWalkSearch
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.tools.drawer.graph_drawer import *

simulator = StateVectorSimulator()
search = QuantumWalkSearch(simulator)
sample = search.run(index_qubits=4, targets=[4], a_r=5 / 8, a_nr=1 / 8)
draw_samples_with_auxiliary(sample, 4, 2, save_path=".")
