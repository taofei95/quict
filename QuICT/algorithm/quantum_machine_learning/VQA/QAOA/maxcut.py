import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import os, sys, time
import random
import torch
from typing import Dict, List, Union

from QuICT.algorithm.quantum_machine_learning.VQA.QAOA.qaoa import QAOA
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian


def maxcut_hamiltonian(edges: Union[List, Dict]):
    pauli_list = []
    for edge in edges:
        pauli_list.append([-1, "Z" + str(edge[0]), "Z" + str(edge[1])])
    hamiltonian = Hamiltonian(pauli_list)

    return hamiltonian


def draw_graph(n, edges):
    G = nx.Graph()
    V = range(n)
    G.add_nodes_from(V)
    G.add_edges_from(edges)

    pos = nx.circular_layout(G)
    options = {
        "with_labels": True,
        "font_size": 20,
        "font_weight": "bold",
        "font_color": "white",
        "node_size": 2000,
        "width": 2,
    }
    nx.draw_networkx(G, pos, **options)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()
    return


def draw(state_list):
    plt.figure()
    # p = state.real.cpu().detach().numpy() * state.real.cpu().detach().numpy()
    # print(sum(p))
    # plt.bar(range(p.shape[0]), p)
    plt.bar(range(len(state_list)), state_list)
    plt.show()


def solve_maxcut(n, edges, p, max_iters, lr, shots=0, plot=False):
    if plot:
        draw_graph(n, edges)
    hamiltonian = maxcut_hamiltonian(edges)
    qaoa = QAOA(n, p, hamiltonian)
    state = qaoa.run(optimizer=torch.optim.Adam, lr=lr, max_iters=max_iters)
    circuit = qaoa.net.construct_circuit()
    return state, circuit, qaoa.net.gamma, qaoa.net.beta


if __name__ == "__main__":

    def seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    seed(17)
    n = 4
    p = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    state, circuit, gamma, beta = solve_maxcut(n, edges, p=p, max_iters=120, lr=0.1)
    from QuICT.simulation.state_vector import ConstantStateVectorSimulator
    simulator = ConstantStateVectorSimulator()
    simulator.vector = state.cpu().detach().numpy()
    simulator.circuit = circuit
    simulator._qubits = circuit.width()
    state_list = simulator.sample(1024)
    print(gamma)
    print(beta)
    print(state)
    print(state_list)
    draw(state_list)

