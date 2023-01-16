import torch
import tqdm
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.algorithm.quantum_machine_learning.model.VQA import QAOANet
from QuICT.algorithm.tools.drawer.graph_drawer import *
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


def maxcut_hamiltonian(edges):
    pauli_list = []
    for edge in edges:
        pauli_list.append([-1.0, "Z" + str(edge[0]), "Z" + str(edge[1])])
    hamiltonian = Hamiltonian(pauli_list)

    return hamiltonian


def get_result(edges, solution_bit):
    cut_edges = []
    for (u, v) in edges:
        if solution_bit[u] != solution_bit[v]:
            cut_edges.append((u, v))

    max_cut_num = len(cut_edges)
    return max_cut_num, cut_edges


# Set graph
n = 5
nodes = list(range(n))
edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 3], [2, 4]]
H = maxcut_hamiltonian(edges)
draw_graph(nodes, edges)

# Set ML related parameters
p = 4
MAX_ITERS = 150
LR = 0.1
SEED = 17
set_seed(SEED)

qaoa_net = QAOANet(n_qubits=n, p=p, hamiltonian=H)
optim = torch.optim.Adam([dict(params=qaoa_net.parameters(), lr=LR)])

# training
qaoa_net.train()
loader = tqdm.trange(MAX_ITERS, desc="Training", leave=True)
for it in loader:
    optim.zero_grad()
    state = qaoa_net()
    loss = qaoa_net.loss_func(state)
    loss.backward()
    optim.step()
    loader.set_postfix(loss=loss.item())

qaoa_cir = qaoa_net.construct_circuit()
simulator = ConstantStateVectorSimulator()
simulator.vector = state.cpu().detach().numpy()
simulator.circuit = qaoa_cir
simulator._qubits = qaoa_cir.width()
prob = simulator.sample(shots=1000)
solution = prob.index(max(prob))
solution_bit = ("{:0" + str(n) + "b}").format(solution)
draw_maxcut_result(nodes, edges, solution_bit)
max_cut_num, cut_edges = get_result(edges, solution_bit)
print("Max cut edges: {}".format(max_cut_num))
print("Cut edges: {}".format(cut_edges))
