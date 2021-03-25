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


from utility import *
from table_based_mcts import *
from random_circuit_generator import *
from RL.nn_model import *
from RL.evaluator import *


if __name__ == "__main__":
    torch.cuda.set_device(2)
    model_path = "./warmup/output/checkpoints/policy_model_state_dict"
    output_path = "./warmup/output"
    evaluator = Evaluator(coupling_graph = "ibmq20", config = default_config)
    evaluator(model_path = model_path, mode = EvaluateMode.PROB, output_path = output_path)

