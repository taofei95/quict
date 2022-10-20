import random
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch

from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.VQA.QAOA import QAOA
from QuICT.algorithm.tools.drawer import graph_drawer as gd
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


class MaxCut:
    def __init__(self, n: int, edges: list, simulator=ConstantStateVectorSimulator()):
        self._n = n
        self._edges = edges
        self.solution_bit = None
        self.model_path = None
        self.simulator = simulator

    def _seed(self, seed: int):
        """Random seed."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

    def _maxcut_hamiltonian(self):
        """Generate the hamiltonian

        Returns:
            Hamiltonian: _description_
        """
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

    def _result(self):
        """Solve the number of maxcut and cut edges according to the state."""
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

    def draw_result(self):
        """Draw the result of MaxCut.
        
           The two node-sets are colored red and blue respectively.
           The red dashed lines represent the cut edges.
        """
        gd.draw_maxcut_result(
            range(self._n), self._edges, self.solution_bit, save_path=self.model_path
        )

    def draw_graph(self):
        """Draw the graph."""
        gd.draw_graph(range(self._n), self._edges)

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
        resume: Union[bool, int] = False,
        device="cuda:0",
    ):
        """_summary_

        Args:
            p (int, optional): _description_. Defaults to 4.
            max_iters (int, optional): _description_. Defaults to 150.
            lr (float, optional): _description_. Defaults to 0.1.
            shots (int, optional): _description_. Defaults to 1000.
            seed (int, optional): _description_. Defaults to 0.
            draw_circuit (bool, optional): _description_. Defaults to False.
            plot_prob (bool, optional): _description_. Defaults to False.
            load_model (_type_, optional): _description_. Defaults to None.
            save_model (bool, optional): _description_. Defaults to False.
            resume (Union[bool, int], optional): _description_. Defaults to False.
            device (str, optional): _description_. Defaults to "cuda:0".

        Returns:
            _type_: _description_
        """
        self._seed(seed)
        hamiltonian = self._maxcut_hamiltonian()
        qaoa = QAOA(self._n, p, hamiltonian, device=device, seed=seed)
        if load_model is not None and resume is False:
            state = qaoa.test(model_path=load_model)
        else:
            state = qaoa.train(
                optimizer="Adam",
                lr=lr,
                max_iters=max_iters,
                model_path=load_model,
                save_model=save_model,
                resume=resume,
            )
        self.model_path = qaoa.model_path
        circuit = qaoa.net.construct_circuit()
        circuit.qasm("qasm5.txt")
        if draw_circuit:
            if self.model_path is None:
                circuit.draw()
            else:
                circuit.draw(filename=self.model_path + "/Maxcut_circuit.jpg")
        self.simulator.vector = state.cpu().detach().numpy()
        self.simulator.circuit = circuit
        self.simulator._qubits = circuit.width()
        prob = self.simulator.sample(shots)
        if plot_prob:
            self._draw_prob(prob, shots)
        solution = prob.index(max(prob))
        self.solution_bit = ("{:0" + str(self._n) + "b}").format(solution)
        max_cut_num, cut_edges = self._result()

        return max_cut_num, cut_edges
