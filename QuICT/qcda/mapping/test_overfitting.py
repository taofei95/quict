import os
import sys
import time
import cProfile
import numpy as np
import logging

import torch.multiprocessing as mp
from torch.multiprocessing import Pool
from tensorboardX import SummaryWriter

from multiprocessing import shared_memory, Lock
from multiprocessing.managers import BaseManager, BaseProxy
from multiprocessing.sharedctypes import Value

import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import List, Sequence, Tuple, Optional, Union, Dict
from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.layout import *


from utility import *
from table_based_mcts import *
from random_circuit_generator import *
from RL.experience_pool_v4 import *
from RL.nn_model import *
from RL.dataloader import *



if __name__ == "__main__":
    dir_path = "/home/shoulifu/QuICT/QuICT/qcda/mapping/"
    input_path = "./policy/input"
    graph_name = "ibmq20"
    config = default_config
    coupling_graph = get_coupling_graph(graph_name)
    experience_pool = DataLoader(max_capacity = config.maximum_capacity, num_of_nodes = config.num_of_nodes, num_of_class = coupling_graph.num_of_edge)
    experience_pool.load_data(input_path)
    mcts = TableBasedMCTS(mode = MCTSMode.TRAIN, coupling_graph = graph_name, experience_pool = experience_pool)   
    num_of_circuits = 5
    
    circuits = []
    qubit_mapping = [i for i in range(coupling_graph.size)]
    for i in range(num_of_circuits):
        qc = OPENQASMInterface.load_file(f"{dir_path}/benchmark/QASM example/{small_benchmark[i]}.qasm")
        circuit =qc.circuit
        circuits.append(circuit)

    for i in range(len(circuits)):
        res = mcts.test_random_simulation(logical_circuit = circuits[i], init_mapping = qubit_mapping ,num_of_gates = -1)
        print(qubit_mapping)
    
    experience_pool.save_data(input_path)


 
