import os
import sys

from typing import List, Tuple, Optional, Union, Dict

from QuICT.tools.interface import *
from QuICT.core.circuit import *
from QuICT.core.layout import *
from mcts import MCTS 

small_benchmark = ("rd84_142",
                #    "adr4_197",
                #    "radd_250",
                #    "z4_268",
                #    "sym6_145",
                #    "misex1_241",
                #    "rd73_252",
                #    "cycle10_2_110",
                #    "square_root_7",
                #    "sqn_258",
                #    "rd84_253",
                   ) 


def test_mapping(input_path: str, output_path: str,  num_of_qubits: int, init_mapping: List[int], graph_name: str):
    qc = OPENQASMInterface.load_file(input_path)
    circuit =qc.circuit

    logical_qubit_num = qc.qbits
    physical_qubit_num = num_of_qubits

    print(input_path)
    mcts = MCTS(graph_name = graph_name)
    mcts.search(logical_circuit =circuit, init_mapping = init_mapping)
    
    print(circuit.circuit_size())
    

if __name__ == "__main__":
    file_path = os.path.realpath(__file__)
    dir_path, file_name = os.path.split(file_path)
    for file_name in small_benchmark:
        input_path = f"{dir_path}/benchmark/QASM example/{file_name}.qasm"
        output_path =  f"{dir_path}/benchmark/QASM example/output/{file_name}.output"
        test_mapping(input_path, output_path, num_of_qubits = 20, init_mapping = [i for i in range(20)], graph_name = "ibmq20")