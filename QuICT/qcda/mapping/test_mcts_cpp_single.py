import os

import sys
import time
from typing import List, Tuple, Optional, Union, Dict

from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.layout import *

from mcts.mcts import MCTS 
from utility.utility import *
from utility.init_mapping import simulated_annealing




    

if __name__ == "__main__":
    file_path = os.path.realpath(__file__)
    dir_path, file_name = os.path.split(file_path)
    initial = 0
    graph_name = "ibmq20"
    gamma = 0.8
    c = 20
    Gsim = 20
    Nsim= 2
    selection_times = 5000
    num_of_swap_gates = 15
    extended = True
    method  = 0
    bench_mark_name = "large"
    repeat_times = 5
    major=4
    mcts = MCTS(graph_name = graph_name, gamma = gamma, c = c, Gsim = Gsim,
                Nsim = Nsim, selection_times = selection_times, num_of_swap_gates = num_of_swap_gates, 
                virtual_loss = 0, bp_mode = 0, num_of_process = 4,
                extended =  extended, with_predictor = False, is_generate_data = False,
                info = 0, method = method, major = major)
  
    file_name = 'hwb8_113.qasm' 
    input_path = f"{dir_path}/benchmark/QASM example/{file_name}"
    
    #init_mapping = [4, 12, 9, 13, 3, 11, 7, 14, 8, 0, 17, 10, 2, 19, 6, 15, 16, 5, 1, 18]
    init_mapping = [i for i in range(20)]
    qc = OPENQASMInterface.load_file(input_path)
    
    circuit =qc.circuit
    logical_qubit_num = qc.qbits

    print(input_path)
    result = []
    for _ in range(repeat_times):
        start = time.time()
        res = mcts.search(logical_circuit =circuit, init_mapping = init_mapping)    
        end = time.time()
        result.append((res, (end-start)))
        print(end-start)
        print(res)
