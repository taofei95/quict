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
    model_path = "./alphaQ/output/checkpoint/model_state_dict_initial"
    output_path = "./alphaQ/output"
    evaluator = Evaluator(coupling_graph = "ibmq20", config = default_config, benchmark = Benchmark.REVLIB)
    evaluator(model_path = model_path, rl_mode  = MCTSMode.SEARCH, mode = EvaluateMode.PROB, sim = SimMode.AVERAGE, output_path = output_path)

