import sys
import tqdm

# from QuICT.algorithm.quantum_algorithm.quantum_walk import numpy_ml_copy
from QuICT.algorithm.quantum_machine_learning.numpy_ml_copy.optimizers.init import *
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.algorithm.tools.drawer.graph_drawer import *
from QuICT.algorithm.quantum_machine_learning.model import QAOA
from QuICT.simulation.state_vector import StateVectorSimulator


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

p = 4  # 量子电路层数
MAX_ITERS = 150  # 最大迭代次数
LR = 0.1  # 梯度下降的学习率
SEED = 17  # 随机数种子
SHOTS = 1000

set_seed(SEED)  # 设置全局随机种子

qaoa_net = QAOA(n_qubits=n, p=p, hamiltonian=H)
optim = Adam(lr=LR)

# 开始训练
loader = tqdm.trange(MAX_ITERS, desc="Training", leave=True)
for it in loader:
    state, loss = qaoa_net.run_step(optim)
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
