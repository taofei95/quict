from typing import List, Dict, Tuple

from multiprocessing import Queue, Pipe
from multiprocessing.connection import Connection

import time 
import torch
import numpy as np

from .nn_model import *
from .rl_based_mcts import *
from QuICT.qcda.mapping.utility import *
from QuICT.qcda.mapping.coupling_graph import *
from QuICT.qcda.mapping.random_circuit_generator import *

small_benchmark = ("rd84_142",
                   "adr4_197",
                   "radd_250",
                   "z4_268",
                   "sym6_145",
                   "misex1_241",
                   "rd73_252",
                   "cycle10_2_110",
                   "square_root_7",
                   "sqn_258",
                   "rd84_253") 

dir_path = "/home/shoulifu/QuICT/QuICT/qcda/mapping/"

class Evaluator(object):
    def __init__(self, coupling_graph: str = None,  config: GNNConfig = None,  max_gates: int = 1000, min_gates: int = 100, max_qubits: int = 20, min_qubits: int = 5, num_of_circuits: int = 5, benchmark: Benchmark = Benchmark.RANDOM):
        self._graph_name = coupling_graph
        self._coupling_graph = get_coupling_graph(coupling_graph)
        self._config = config
        self._model = SequenceModel(n_qubits = self._coupling_graph.size, n_class = self._coupling_graph.num_of_edge,  config = self._config).to(config.device).float()
        self._counter_model = SequenceModel(n_qubits = self._coupling_graph.size, n_class = self._coupling_graph.num_of_edge,  config = self._config).to(config.device).float()

        self._radndom_circuit_generator = RandomCircuitGenerator(max_num_of_qubits = max_qubits, min_num_of_qubits = min_qubits, minimum = min_gates, maximum = max_gates, seed = int(time.time()))
        self._circuits =[]
        if benchmark  == Benchmark.RANDOM:
            for _ in range(num_of_circuits):
                self._circuits.append(self._radndom_circuit_generator())
        elif benchmark == Benchmark.REVLIB:
            for i in range(num_of_circuits):
                qc = OPENQASMInterface.load_file(f"{dir_path}/benchmark/QASM example/{small_benchmark[i]}.qasm")
                circuit =qc.circuit
                self._circuits.append(circuit)
        self._res = []

    def __call__(self, model_path: str = None, counter_model_path: str = None, rl_mode: MCTSMode = MCTSMode.SEARCH, mode: EvaluateMode = EvaluateMode.SEARCH, sim: SimMode = SimMode.MAX, output_path: str = None):
        self._model.load_state_dict(torch.load(model_path))
        self._counter_model.load_state_dict(torch.load(counter_model_path))

        mcts = RLBasedMCTS(mode = rl_mode, device = self._config.device, model = self._model, coupling_graph = self._graph_name)
        counter_mcts = RLBasedMCTS(mode = rl_mode, device = self._config.device, model = self._counter_model, coupling_graph = self._graph_name)
        init_mapping = [i for i in range(self._coupling_graph.size)]

        with open(f"{output_path}",'w') as f:
            for i, circuit in enumerate(self._circuits):
                if mode == EvaluateMode.SEARCH:
                    res = mcts.search(logical_circuit = circuit, init_mapping = init_mapping)
                    self._res.append(res)
                    f.write("%d  %d \n"%(res[0], res[1]))
                
                elif mode == EvaluateMode.PROB:
                    
                    num_of_swap_gates_random= 0
                    num_of_swap_gates_counter = counter_mcts.random_simulate(logical_circuit = circuit, init_mapping = init_mapping)
                    num_of_swap_gates = mcts.random_simulate(logical_circuit = circuit, init_mapping = init_mapping)
                    #num_of_swap_gates_random = mcts.test_random_simulation(logical_circuit = circuit, init_mapping = init_mapping, num_of_gates = -1)
                    self._res.append(num_of_swap_gates)
                    f.write("%d %d %d %d\n"%(circuit.circuit_size(), num_of_swap_gates, num_of_swap_gates_counter, num_of_swap_gates_random))
                
                elif mode == EvaluateMode.EXTENDED_PROB:
                    num_of_swap_gates_nn =  mcts.extended_random_simulate(logical_circuit = circuit, init_mapping = init_mapping)
                    num_of_swap_gates = 0
                    #num_of_swap_gates = mcts.random_simulate(logical_circuit = circuit, init_mapping = init_mapping)
                    #num_of_swap_gates_random = mcts.test_random_simulation(logical_circuit = circuit, init_mapping = init_mapping, num_of_gates = -1)
                    self._res.append(num_of_swap_gates_nn)
                    f.write("%d %d %d \n"%(circuit.circuit_size() ,num_of_swap_gates, num_of_swap_gates_nn))
                
                else:
                    raise Exception("No such mode.")

    


            



