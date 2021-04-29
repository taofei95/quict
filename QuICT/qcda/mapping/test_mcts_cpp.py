import os

import sys
import time
from typing import List, Tuple, Optional, Union, Dict

from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.layout import *

from QuICT.qcda.mapping.mcts.mcts import MCTS 
from QuICT.qcda.mapping.utility.utility import *
from init_mapping import simulated_annealing




    

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
    log_path =  f"{dir_path}/benchmark/log/mcts_cpp_{bench_mark_name}_{major}_{gamma}_{c}_{Gsim}_{Nsim}_{selection_times}_{initial}_{num_of_swap_gates}_{extended}_{method}.log"
    with open(log_path, "w") as f:
        for  file_name, init_mapping in zip(bench_mark[bench_mark_name],initial_map_lists[bench_mark_name]):
            input_path = f"{dir_path}/benchmark/QASM example/{file_name}"
            if initial == 1:
                init_mapping = [i for i in range(20)]
            else:
                init_mapping = init_mapping


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

            print(",".join(
                [file_name, str(result[0][0][0]),
                ",".join(
                        [ ",".join([str(result[i][0][1]), str(result[i][1])]) for i in range(repeat_times)])
                ]),
                file = f)