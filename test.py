import numpy as np
import time
import networkx as nx
from Qcover.core import Qcover
from Qcover.backends import CircuitByQulacs
from Qcover.optimizers import COBYLA

p = 1
g = nx.Graph()
nodes = [(0, 0), (1, 0), (2, 0), (3, 0)]
edges = [(0, 1, 1), (1, 2, 1), (2, 3, 1), (3, 0, 1)]

for nd in nodes:
    u, w = nd[0], nd[1]
    g.add_node(int(u), weight=int(w))
for ed in edges:
    u, v, w = ed[0], ed[1], ed[2]
    g.add_edge(int(u), int(v), weight=int(w))


optc = COBYLA(options={'tol': 1e-3, 'disp': True})
qulacs_bc = CircuitByQulacs()
qc = Qcover(g, p,
            # research_obj="QAOA",
            optimizer=optc,  #@ optc,
            backend=qulacs_bc)  #qiskit_bc, qt, , cirq_bc, projectq_bc

st = time.time()
sol = qc.run(is_parallel=False)  #True
ed = time.time()
print("time cost by QAOA is:", ed - st)
print("solution is:", sol)
params = sol["Optimal parameter value"]
qc.backend._pargs = params
out_count = qc.backend.get_result_counts(params)
res_exp = qc.backend.expectation_calculation()
print("the optimal expectation is: ", res_exp)
qc.backend.sampling_visualization(out_count)