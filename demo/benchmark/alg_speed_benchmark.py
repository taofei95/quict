import os
import random
import time
import numpy as np
import torch

from QuICT.algorithm.quantum_algorithm import ShorFactor, Grover, QuantumWalkSearch
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *
from QuICT.core.gate.backend.mct.mct_one_aux import MCTOneAux
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.algorithm.quantum_machine_learning.utils import *
from QuICT.algorithm.quantum_machine_learning.model.VQA import QAOANet
from QuICT.algorithm.tools.drawer.graph_drawer import *
from QuICT.algorithm.quantum_machine_learning.ansatz_library import QNNLayer
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
import tqdm

from unit_test.algorithm.quantum_algorithm.grover_unit_test import main_oracle


def shor_benchmark():
    run_test_modes = ["BEA", "HRS", "BEA_zip", "HRS_zip"]
    number_list = [15, 21, 30, 35]
    data = open("alg_shor_benchmark_data1.txt", "w+")
    for mode in run_test_modes:
        data.write(f"mode: {mode}\n")
        for number in number_list:
            print("Shor: number: {}".format(number))
            stime = time.time()
            a = ShorFactor(mode=mode, simulator=ConstantStateVectorSimulator()).run(N=number, forced_quantum_approach=True)
            ttime = time.time()
            cir = ShorFactor(mode=mode).circuit(N=number)[0]
            print(cir)
            data.write(f"Quict time : {round(ttime - stime, 4)}\n")
            data.write(f"qubits: {number}, circuit qubits:{cir.width()} \n")

    data.close()


def grover_benchmark():
    data = open("alg_grover_benchmark_data.txt", "w+")

    def main_oracle(n, f):
        result_q = [n]
        cgate = CompositeGate()
        target_binary = bin(f[0])[2:].rjust(n, "0")
        with cgate:
            # |-> in result_q
            X & result_q[0]
            H & result_q[0]
            # prepare for MCT
            for i in range(n):
                if target_binary[i] == "0":
                    X & i
        MCTOneAux().execute(n + 2) | cgate
        # un-compute
        with cgate:
            for i in range(n):
                if target_binary[i] == "0":
                    X & i
            H & result_q[0]
            X & result_q[0]
        return 2, cgate

    for n in range(3, 7):
        data.write(f"{n} \n")
        error = 0
        N = 2**n
        target = random.randint(0, N)
        f = [target]
        k, oracle = main_oracle(n, f)
        begin_time = time.time()
        grover = Grover(simulator=ConstantStateVectorSimulator())
        result = grover.run(n, k, oracle)
        last_time = time.time()
        data.write(f"speed:{round(last_time - begin_time, 4)} \n")

    data.close()


def quantum_walk_search_benchmark():
    data = open("alg_quantum_walk_search_benchmark_data.txt", "w+")
    simulator = ConstantStateVectorSimulator()
    index_qubits_list = [3, 4, 5, 6, 8]
    for index in index_qubits_list:
        running_time = 0
        for i in range(10):
            begin_time = time.time()
            search = QuantumWalkSearch(simulator)
            sample = search.run(index_qubits=index, targets=[4], a_r=5 / 8, a_nr=1 / 8)
            last_time = time.time()
            running_time += last_time - begin_time
        data.write(f"qubits{index} speed: {round(running_time / 10, 4)}")
        # draw_samples_with_auxiliary(sample, index, int(np.ceil(np.log2(index))))

    data.close()


def maxcut_benchmark():
    data = open("alg_maxcut_benchmark_data2.txt", "w+")

    def maxcut_hamiltonian(edges):
        pauli_list = []
        for edge in edges:
            pauli_list.append([-1.0, "Z" + str(edge[0]), "Z" + str(edge[1])])
        hamiltonian = Hamiltonian(pauli_list)
        return hamiltonian

    def random_edges(n_nodes, n_edges):
        assert n_nodes >= 2
        assert n_edges <= n_nodes * (n_nodes - 1) / 2
        edges = []
        nodes = np.arange(n_nodes)
        while len(edges) < n_edges:
            edge = list(np.random.choice(nodes, 2, replace=False))
            if edge in edges or edge[::-1] in edges:
                continue
            edges.append(edge)
        return edges

    device = torch.device("cuda:0")
    p = 4  # 量子电路层数
    MAX_ITERS = 100  # 最大迭代次数
    LR = 0.1  # 梯度下降的学习率
    SEED = 0  # 随机数种子
    set_seed(SEED)  # 设置全局随机种子

    n_node_list = [4, 5, 6, 8, 10]
    n_edge_list = [4, 6, 8, 11, 15]

    for (n_node, n_edges) in zip(n_node_list, n_edge_list):
        data.write(f"nodes: {n_node}, edges: {n_edges} \n")
        edges = random_edges(n_node, n_edges)
        H = maxcut_hamiltonian(edges)
        qaoa_net = QAOANet(n_qubits=n_node, p=p, hamiltonian=H)
        optim = torch.optim.Adam([dict(params=qaoa_net.parameters(), lr=LR)])

        # training
        begin_time = time.time()
        qaoa_net.train()
        loader = tqdm.trange(MAX_ITERS, desc="Training", leave=True)
        for it in loader:
            optim.zero_grad()
            state = qaoa_net()
            loss = qaoa_net.loss_func(state)
            loss.backward()
            optim.step()
            loader.set_postfix(loss=loss.item())
        last_time = time.time()
        circuit = qaoa_net.construct_circuit()
        simulator = ConstantStateVectorSimulator()
        simulator.vector = state.cpu().detach().numpy()
        simulator.circuit = circuit
        simulator._qubits = circuit.width()
        prob = simulator.sample(shots=1000)
        solution = prob.index(max(prob))
        solution_bit = ("{:0" + str(n_node) + "b}").format(solution)
        draw_maxcut_result(list(range(n_node)), edges, solution_bit)
        data.write(f"speed {round(last_time - begin_time, 4)} \n")
        data.write(
            f"circuit width:{circuit.width()}, circuit size:{circuit.size()}, circuit depth:{circuit.depth()} \n"
        )
    data.close()


def qnn_benchmark():
    data = open("alg_qnn_benchmark_data.txt", "w+")
    img_size = [4, 4]
    n_data_qubits = img_size[0] * img_size[1]
    data_qubits = list(range(n_data_qubits))
    result_qubit = n_data_qubits
    device = torch.device("cuda:0")
    layers = ["XX", "ZZ"]

    params = torch.rand(len(layers), n_data_qubits).to(device)
    n_qubits = n_data_qubits + 1

    # model circuit
    begin_time = time.time()
    pqc = QNNLayer(data_qubits, result_qubit, device)
    model_circuit = Circuit(n_qubits)
    X | model_circuit(data_qubits)
    H | model_circuit(data_qubits)
    sub_circuit = pqc.circuit_layer(layers, params)
    model_circuit.extend(sub_circuit.gates)
    H | model_circuit(data_qubits)

    # data circuit
    img = np.random.randint(0, 2, n_data_qubits)
    data_circuit = Circuit(n_qubits)
    for i in data_qubits:
        if img[i] > 0.5:
            X | data_circuit(i)

    # final circuit
    circuit = data_circuit
    circuit.extend(model_circuit.gates)
    last_time = time.time()
    data.write(f"run time: {round(last_time - begin_time, 4)}")
    data.write(
        f"circuit width:{circuit.width()}, circuit size:{circuit.size()}, circuit depth:{circuit.depth()}"
    )


if __name__ == "__main__":
    # print("Running Grover")
    # grover_benchmark()
    # print("Running Quantum walk search")
    # quantum_walk_search_benchmark()
    print("Running Shor")
    shor_benchmark()
    # print("Running MaxCut")
    # maxcut_benchmark()
    # qnn_benchmark()
