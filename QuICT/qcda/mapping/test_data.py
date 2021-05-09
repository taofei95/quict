import os
import sys

import numpy as np

from typing import List, Tuple, Optional, Union, Dict

from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.layout import *
from utility.coupling_graph import *
from utility.random_circuit_generator import RandomCircuitGenerator
from utility.dag import * 
from utility.utility import *

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
                    "rd84_253",
                   ) 

mcts_benchmark = (
"hwb5_53",
"radd_250", 
"rd73_252", 
"cycle10_2_110",
"hwb6_56", 
"cm85a_209", 
"rd84_253",
"root_255", 
"mlp4_245", 
"urf2_277", 
"sym9_148",
"hwb7_59", 
"clip_206", 
"sym9_193", 
"dist_223",
"sao2_257", 
"urf5_280",
"urf1_278", 
"sym10_262", 
"hwb8_113", 
)

if __name__ == "__main__":
    file_path = os.path.realpath(__file__)
    dir_path, _ = os.path.split(file_path)
    
    graph_name = "ibmq20"
    file_name = small_benchmark[0]
    coupling_graph = get_coupling_graph(graph_name = graph_name)
    input_path = f"{dir_path}/benchmark/QASM example/hwb7_59.qasm"
    output_dir = f"{dir_path}/cpp_test_data"
    random_circuit_generator = RandomCircuitGenerator(minimum = 500, maximum = 1500)

    qc = OPENQASMInterface.load_file(input_path) 
    circuit =qc.circuit
    #circuit = random_circuit_generator()
    circuit_dag = DAG(circuit = circuit, mode = Mode.TWO_QUBIT_CIRCUIT)
  
    
    coupling_graph.adj_matrix.tofile(f"{output_dir}/adj_matrix.txt")
    
    coupling_graph.distance_matrix.tofile(f"{output_dir}/distance_matrix.txt")
    coupling_graph.label_matrix.tofile(f"{output_dir}/label_matrix.txt") 
    coupling_graph.node_feature.tofile(f"{output_dir}/feature_matrix.txt") 

    init_mapping = np.array([14, 19, 13, 8, 3, 2, 17, 1, 15, 9, 18, 16, 5, 11, 10, 4, 6, 12, 7, 0], dtype = np.int32)
    qubit_mask = np.zeros(coupling_graph.size, dtype = np.int32) -1
     
    for i, qubit in enumerate(circuit_dag.initial_qubit_mask):
        qubit_mask[init_mapping[i]] = qubit 
        
    front_layer_ = np.array([ circuit_dag.index[i]  for i in circuit_dag.front_layer ], dtype = np.int32)
    qubit_mask_ =  np.array([ circuit_dag.index[i] if i != -1 else -1 for i in qubit_mask ],  dtype = np.int32)


    circuit_dag.node_qubits.tofile(f"{output_dir}/circuit.txt") 
    circuit_dag.compact_dag.tofile(f"{output_dir}/dependency_graph.txt")

    init_mapping.tofile(f"{output_dir}/qubit_mapping.txt") 
    front_layer_.tofile(f"{output_dir}/front_layer.txt")
    qubit_mask_.tofile(f"{output_dir}/qubit_mask.txt")

    with open(f"{output_dir}/metadata.txt", "w") as f:
        print(coupling_graph.size, file = f)
        print(coupling_graph.num_of_edge, file = f)
        print(coupling_graph.node_feature.shape[1], file = f)
        print(circuit_dag.size, file = f)
        print(front_layer_.shape[0], file = f)