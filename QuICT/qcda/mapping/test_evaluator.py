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
    model_path = "./alphaQ/output/checkpoint/model_state_dict_test_small_data_no_initial"
    counter_model_path = "./alphaQ/output/checkpoint/model_state_dict_initial"
    output_path = "./alphaQ/output/prob/evaluate_res_rl_counter"
    
    evaluator = Evaluator(coupling_graph = "ibmq20", num_of_circuits = 20, config = default_config, benchmark = Benchmark.RANDOM)
    evaluator(model_path = model_path, counter_model_path = counter_model_path, rl_mode  = MCTSMode.SEARCH, mode = EvaluateMode.SEARCH, sim = SimMode.AVERAGE, output_path = output_path)

