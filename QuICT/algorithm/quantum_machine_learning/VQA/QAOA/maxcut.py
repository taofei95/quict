import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import random

from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.VQA.QAOA.qaoa import QAOA
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian


class MaxCut:
    def __init__(self, n: int, edges: list):
        self._n = n
        self._edges = edges
        self.solution_bit = None

    def _seed(self, seed: int):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def _maxcut_hamiltonian(self):
        pauli_list = []
        for edge in self._edges:
            pauli_list.append([-1.0, "Z" + str(edge[0]), "Z" + str(edge[1])])
        hamiltonian = Hamiltonian(pauli_list)

        return hamiltonian

    def _draw_prob(self, prob, shots):
        plt.figure()
        plt.xlabel("Qubit States")
        plt.xlabel("Probabilities")
        plt.bar(range(len(prob)), np.array(prob) / shots)
        plt.show()

    def draw_graph(self):
        plt.figure()
        plt.title("Graph")
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
        plt.title("The result of MaxCut")
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
                if (
                    (u, v) in self._edges
                    or (v, u) in self._edges
                    or [u, v] in self._edges
                    or [v, u] in self._edges
                ):
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

    def _result(self):
        cut_edges = []
        for u in range(self._n):
            for v in range(u + 1, self._n):
                if (
                    (u, v) in self._edges
                    or (v, u) in self._edges
                    or [u, v] in self._edges
                    or [v, u] in self._edges
                ):
                    if self.solution_bit[u] != self.solution_bit[v]:
                        cut_edges.append((u, v))

        max_cut_num = len(cut_edges)
        return max_cut_num, cut_edges

    def solve_maxcut(
        self,
        p: int,
        max_iters: int,
        lr: float,
        shots: int = 1000,
        seed: int = 0,
        draw_circuit=False,
        plot_prob=False,
        device=torch.device("cuda:0"),
    ):
        self._seed(seed)
        hamiltonian = self._maxcut_hamiltonian()
        qaoa = QAOA(self._n, p, hamiltonian, device=device)
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
            self._draw_prob(prob, shots)
        solution = prob.index(max(prob))
        self.solution_bit = ("{:0" + str(self._n) + "b}").format(solution)
        max_cut_num, cut_edges = self._result()

        return max_cut_num, cut_edges


if __name__ == "__main__":
    n = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (1, 3)]
    maxcut = MaxCut(n, edges)
    # maxcut.draw_graph()
    max_cut_num, cut_edges = maxcut.solve_maxcut(
        p=4, max_iters=100, lr=0.1, plot_prob=True, draw_circuit=True
    )
    print("Max cut: {}".format(max_cut_num))
    print("Cut edges: {}".format(cut_edges))
    maxcut.draw_result()
