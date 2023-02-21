from QuICT.algorithm.quantum_algorithm import QuantumWalk
from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.algorithm.tools.drawer.graph_drawer import *

simulator = StateVectorSimulator()
qw = QuantumWalk(simulator)
step = 10
edges = [[1, 3], [2, 0], [3, 1], [0, 2]]
sample = qw.run(step=step, position=4, edges=edges, coin_operator=H.matrix)
print(sample)
draw_samples_with_auxiliary(sample, 2, 1)
