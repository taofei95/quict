import tqdm
import numpy_ml

from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.algorithm.tools.drawer.graph_drawer import *
from QuICT.algorithm.quantum_machine_learning.model import QAOA

n = 5
nodes = list(range(n))
edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 3], [2, 4]]


def maxcut_hamiltonian(edges):
    pauli_list = []
    for edge in edges:
        pauli_list.append([-1.0, "Z" + str(edge[0]), "Z" + str(edge[1])])
    hamiltonian = Hamiltonian(pauli_list)

    return hamiltonian


H = maxcut_hamiltonian(edges)

p = 4
MAX_ITERS = 150
LR = 0.1
SEED = 1
SHOTS = 1000
set_seed(SEED)

optim = numpy_ml.neural_nets.optimizers.Adam(lr=LR)
qaoa_net = QAOA(n_qubits=n, p=p, hamiltonian=H, optimizer=optim, device="CPU")

loader = tqdm.trange(MAX_ITERS, desc="Training", leave=True)
for it in loader:
    loss = qaoa_net.run()
    loader.set_postfix(loss=loss.item())

prob = qaoa_net.sample(SHOTS)
plt.figure()
plt.xlabel("Qubit States")
plt.ylabel("Probabilities")
plt.bar(range(len(prob)), np.array(prob) / SHOTS)
plt.show()

solution = prob.index(max(prob))
solution_bit = ("{:0" + str(n) + "b}").format(solution)
print(solution_bit)

draw_maxcut_result(nodes, edges, solution_bit)
