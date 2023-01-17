import os
import time

from QuICT.core.circuit.circuit import Circuit
from QuICT.algorithm.quantum_algorithm import ShorFactor
from QuICT.algorithm.quantum_algorithm.CNF.cnf import CNFSATOracle
from QuICT.algorithm.quantum_algorithm import Grover
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.core.gate.backend import MCTOneAux
from QuICT.algorithm.quantum_algorithm import QuantumWalkSearch
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
import torch
import tqdm
from QuICT.algorithm.quantum_machine_learning.utils import Hamiltonian
from QuICT.algorithm.quantum_machine_learning.utils.ml_utils import *
from QuICT.algorithm.quantum_machine_learning.model.VQA import QAOANet
from QuICT.algorithm.tools.drawer.graph_drawer import *
from QuICT.simulation.state_vector import ConstantStateVectorSimulator

def shor_benchmark():
    run_test_modes = ["BEA", "HRS", "BEA_zip", "HRS_zip"]
    number_list = [15, 21, 33, 35]
    data = open("alg_shor_benchmark_data.txt", "w+")
    for mode in run_test_modes:
        data.write(f"mode: {mode}\n")
        for number in number_list:
            stime = time.time()
            a = ShorFactor(mode=mode).run(N=number, forced_quantum_approach=True)
            ttime = time.time()
            cir = ShorFactor(mode=mode).circuit(N=number)[0]
            data.write(f"Quict time : {round(ttime - stime, 4)}\n")
            data.write(f"qubits: {number}, circuit qubits:{cir.width()} \n")

    data.close()

def grover_benchmark():
    data = open("alg_grover_benchmark_data.txt", "w+")
    n,f =
    result_q = [n]
    cgate = CompositeGate()
    target_binary = bin(f[0])[2:].rjust(n, "0")
    with cgate:
        X & result_q[0]
        H & result_q[0]
        for i in range(n):
            if target_binary[i] == "0":
                X & i
    MCTOneAux().execute(n + 2) | cgate
    with cgate:
        for i in range(n):
            if target_binary[i] == "0":
                X & i
        H & result_q[0]
        X & result_q[0]

    n = 4
    target = 0b0110
    f = [target]
    k, oracle = main_oracle(n, f)
    grover = Grover(simulator=ConstantStateVectorSimulator())
    result = grover.run(n, k, oracle)
    print(result)
   

    data.close()
    
def cnf_benchmark():
    data = open("alg_cnf_benchmark_data.txt", "w+")

    file_name = os.listdir("QuICT/benchmark/test_data/cnf_file")
    for file in file_name:
        file_name_test = os.path.join(os.path.dirname(os.path.dirname(__file__))) + f"/test_data/cnf_file/{file}"
        n_var, n_aux = 3, 3
        data.write(f"{file} \n")
        begin_time = time.time()
        cnf = CNFSATOracle()
        cnf.run(file_name_test, n_aux, 1)
        last_time = time.time()
        data.write(f"speed:{round(last_time - begin_time, 4)} \n")
        cir = Circuit(n_var + n_aux + 1)
        cnf.circuit() | cir
        data.write(f"circuit width{cir.width()}size{cir.size()}depth{cir.depth()} \n")

    data.close()

def quantum_walk_search_benchmark():
    data = open("alg_quantum_walk_search_benchmark_data.txt", "w+")
    simulator = ConstantStateVectorSimulator()
    index_qubits_list = [4, 8, 16, 32]
    for index in index_qubits_list:
        begin_time = time.time
        search = QuantumWalkSearch(simulator)
        sample = search.run(index_qubits=index, targets=[4], a_r=5 / 8, a_nr=1 / 8)
        last_time = time.time
        data.write(f"speed: {round(last_time - begin_time, 4)}")

    data.close()
    
def maxcut_benchmark():
    data = open("alg_maxcut_benchmark_data.txt", "w+")
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
    max_cut_num, cut_edges = get_result(edges, solution_bit)
    data.write(f"'Max cut edges: {}'.format(max_cut_num) \n")
    print("Max cut edges: {}".format(max_cut_num))
    print("Cut edges: {}".format(cut_edges))
    

    data.close()
    
if __name__ == '__main__':
    # shor_benchmark()
    cnf_benchmark()