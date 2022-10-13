import numpy as np
import torch
import random
import networkx as nx
import matplotlib.pyplot as plt
from typing import Union

from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.VQA.QAOA.qaoa import QAOA
from QuICT.algorithm.quantum_machine_learning.utils.hamiltonian import Hamiltonian


class MaxCut:
    def __init__(self, n: int, edges: list):
        self._n = n
        self._edges = edges
        self.solution_bit = None
        self.model_path = None

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
        plt.ylabel("Probabilities")
        plt.bar(range(len(prob)), np.array(prob) / shots)
        if self.model_path is not None:
            plt.savefig(self.model_path + "/Probabilities.jpg")
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
        if self.model_path is not None:
            plt.savefig(self.model_path + "/Maxcut_result.jpg")
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
        p: int = 4,
        max_iters: int = 150,
        lr: float = 0.1,
        shots: int = 1000,
        seed: int = 0,
        draw_circuit=False,
        plot_prob=False,
        load_model=None,
        save_model=False,
        resume: Union[bool, list] = False,
        device=torch.device("cuda:0"),
    ):
        self._seed(seed)
        hamiltonian = self._maxcut_hamiltonian()
        qaoa = QAOA(self._n, p, hamiltonian, device=device)
        if load_model is not None and load_model != "" and resume is False:
            state = qaoa.test(model_path=load_model)
        else:
            state = qaoa.train(
                optimizer=torch.optim.Adam,
                lr=lr,
                max_iters=max_iters,
                model_path=load_model,
                save_model=save_model,
                resume=resume,
            )
        self.model_path = qaoa.model_path
        circuit = qaoa.net.construct_circuit()
        if draw_circuit:
            if self.model_path is None:
                circuit.draw()
            else:
                circuit.draw(filename=self.model_path + "/Maxcut_circuit.jpg")
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

    def circle_graph(n):
        edges = []
        for i in range(n - 1):
            edges.append((i, i + 1))
        edges.append((n - 1, 0))
        return edges

    n = 4
    maxcut = MaxCut(n, circle_graph(n))
    # training
    max_cut_num, cut_edges = maxcut.solve_maxcut(
        p=n,
        max_iters=100,
        lr=0.1,
        plot_prob=True,
        save_model=True,
        # load_model="QAOA_model_2022-10-13-13_58_36"
    )
    print("Max cut: {}".format(max_cut_num))
    print("Cut edges: {}".format(cut_edges))
    maxcut.draw_result()
