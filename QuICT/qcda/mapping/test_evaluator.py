import os
import sys
import time

import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import List, Tuple, Optional, Union, Dict
from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.layout import *

from RL.evaluator import *


if __name__ == "__main__":
    torch.cuda.set_device(0)
    #model_path = "./policy/output/checkpoints/policy_model_state_dict_ctexm"
    #output_path = "./policy/output"
    #counter_model_path = "./policy/output/checkpoints/model_state_dict_test_overfitting"
    counter_model_path = "./alphaQ/output/checkpoint/model_state_dict_test_small_data_extended"
    model_path = "./alphaQ/output/checkpoint/model_state_dict_test_small_data_extended_1"
    output_path = "./policy/output/evaluate_res_rl_extended_data_1"
    alpha_config = GNNConfig(maximum_capacity = 200000, num_of_gates = 150, maximum_circuit = 1500, minimum_circuit = 200, batch_size = 128, ff_hidden_size = 128, num_self_att_layers=4, dropout = 0.5, value_head_size = 128,
                       gamma = 0.7, num_U2GNN_layers=2, learning_rate = 0.001, weight_decay = 1e-4, num_of_epochs = 50, device = torch.device("cuda"), graph_name = 'ibmq20',num_of_process = 64, feature_update = True, gat = False, 
                       mcts_c = 20, loss_c = 10, n_gat = 2)
    
    evaluator = Evaluator(coupling_graph = "ibmq20", num_of_circuits = 10, config = alpha_config, benchmark = Benchmark.REVLIB)
    evaluator(model_path = model_path, counter_model_path = counter_model_path, rl_mode  = MCTSMode.SEARCH, mode = EvaluateMode.SEARCH, sim = SimMode.AVERAGE, output_path = output_path)

