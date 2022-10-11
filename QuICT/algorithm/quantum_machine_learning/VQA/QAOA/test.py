import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from pandas import cut
import torch
from typing import Dict, List, Union

from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.VQA.QAOA.qaoa import QAOA
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian


class MaxCut:
    def __init__(self, n, edges):
        self._n = n
        self._edges = edges
        self.solution_bit = None

    def maxcut_hamiltonian(self):
        pauli_list = []
        for edge in self._edges:
            pauli_list.append([-1, "Z" + str(edge[0]), "Z" + str(edge[1])])
        hamiltonian = Hamiltonian(pauli_list)

        return hamiltonian

    def draw_prob(self, prob, shots):
        plt.figure()
        plt.bar(range(len(prob)), prob)
        plt.show()

    def draw_graph(self):
        plt.figure()
        G = nx.Graph()
        G.add_nodes_from(range(self._n))
        G.add_edges_from(self._edges)
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

    def draw_result(self):
        plt.figure()
        G = nx.Graph()
        G.add_nodes_from(range(self._n))
        G.add_edges_from(self._edges)
        pos = nx.circular_layout(G)

        node_color = [
            "red" if self.solution_bit[v] == "1" else "#1f78b4" for v in range(self._n)
        ]
        edge_color = []
        edge_style = []
        for u in range(self._n):
            for v in range(u + 1, self._n):
                if (u, v) in self._edges or (v, u) in self._edges:
                    if self.solution_bit[u] == self.solution_bit[v]:
                        edge_color.append("black")
                        edge_style.append("solid")
                    else:
                        edge_color.append("red")
                        edge_style.append("dashed")

        options = {
            "with_labels": True,
            "font_size": 20,
            "font_weight": "bold",
            "font_color": "white",
            "node_size": 2000,
            "width": 2,
        }
        nx.draw(
            G,
            pos,
            node_color=node_color,
            edge_color=edge_color,
            style=edge_style,
            **options
        )
        ax = plt.gca()
        ax.margins(0.20)
        plt.axis("off")
        plt.show()

    def result(self):
        cut_edges = []
        for u in range(self._n):
            for v in range(u + 1, self._n):
                if (u, v) in self._edges or (v, u) in self._edges:
                    if self.solution_bit[u] != self.solution_bit[v]:
                        cut_edges.append((u, v))

        max_cut_num = len(cut_edges)
        return max_cut_num, cut_edges

    def solve_maxcut(
        self, p, max_iters, lr, shots=1000, draw_circuit=False, plot_prob=False
    ):
        hamiltonian = self.maxcut_hamiltonian()
        qaoa = QAOA(self._n, p, hamiltonian)
        state = qaoa.run(optimizer=torch.optim.Adam, lr=lr, max_iters=max_iters)
        circuit = qaoa.net.construct_circuit()
        if draw_circuit:
            circuit.draw()
        simulator = ConstantStateVectorSimulator()
        simulator.vector = state.cpu().detach().numpy()
        simulator.circuit = circuit
        simulator._qubits = circuit.width()
        prob = simulator.sample(shots)
        if plot_prob:
            self.draw_prob(prob, shots)
        solution = prob.index(max(prob))
        self.solution_bit = ("{:0" + str(self._n) + "b}").format(solution)
        max_cut_num, cut_edges = self.result()

        return  # max_cut_num, cut_edges


if __name__ == "__main__":
    n = 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    maxcut = MaxCut(n, edges)
    maxcut.draw_graph()
    max_cut_num, cut_edges = maxcut.solve_maxcut(
        p=4, max_iters=120, lr=0.1, plot_prob=True
    )
    print("Max cut: {}".format(max_cut_num))
    print("Cut edges: {}".format(cut_edges))
    maxcut.draw_result()
