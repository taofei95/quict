from table_based_mcts import MCTSMode
from typing import List, Dict, Tuple

from multiprocessing import Queue, Pipe
from multiprocessing.connection import Connection

import torch
import numpy as np

from .nn_model import *
from .rl_based_mcts import *
from QuICT.qcda.mapping.utility import *
from QuICT.qcda.mapping.coupling_graph import *
from QuICT.qcda.mapping.random_circuit_generator import *

class EvaluateMode(enumerate):
    SEARCH = 0
    EXTENDED_PROB = 1
    PROB = 2

class Evaluator(object):
    def __init__(self, coupling_graph: str = None, config: GNNConfig = None,  max_gates: int = 200, min_gates: int = 50, max_qubits: int = 20, min_qubits: int = 5, num_of_circuits: int = 5):
        self._graph_name = coupling_graph
        self._coupling_graph = get_coupling_graph(coupling_graph)
        self._config = config
        self._model =  self._model = TransformerU2GNN(feature_dim_size = self._coupling_graph.node_feature.shape[1]*2, 
                                        num_classes = self._coupling_graph.num_of_edge, 
                                        config = config).to(config.device).float()
        self._radndom_circuit_generator = RandomCircuitGenerator(max_num_of_qubits = max_qubits, min_num_of_qubits = min_qubits, minimum = min_gates, maximum = max_gates)
        self._circuits =[]
        
        for _ in range(num_of_circuits):
            self._circuits.append(self._radndom_circuit_generator())
        self._res = []

    def __call__(self, model_path: str = None, mode: EvaluateMode = EvaluateMode.SEARCH, output_path: str = None):
        self._model.load_state_dict(torch.load(model_path))
        self._model.eval()
        mcts = RLBasedMCTS(mode = MCTSMode.EVALUATE, device = self._config.device, model = self._model, coupling_graph = self._graph_name)
        init_mapping = [i for i in range(self._coupling_graph.size)]

        with open(f"{output_path}/evaluate_res",'w') as f:
            for i, circuit in enumerate(self._circuits):
                if mode == EvaluateMode.SEARCH:
                    res = mcts.search(logical_circuit = circuit, init_mapping = init_mapping)
                    self._res.append(res)
                    f.write("%d  %d \n"%(res[0], res[1]))
                elif mode == EvaluateMode.PROB:
                    num_of_swap_gates = 0
                    num_of_swap_gates = mcts.random_simulate(logical_circuit = circuit, init_mapping = init_mapping)
                    num_of_swap_gates_random = mcts.test_random_simulation(logical_circuit = circuit, init_mapping = init_mapping, num_of_gates = -1)
                    self._res.append(num_of_swap_gates)
                    f.write("%d %d %d \n"%(circuit.circuit_size(), num_of_swap_gates, num_of_swap_gates_random[0]))
                elif mode == EvaluateMode.EXTENDED_PROB:
                    num_of_swap_gates =  mcts.extended_random_simulate(logical_circuit = circuit, init_mapping = init_mapping)
                    self._res.append(num_of_swap_gates)
                    f.write("%d \n"%(num_of_swap_gates))
                else:
                    raise Exception("No such mode.")

    


            



