import os
from QuICT.core.gate import GateType
from QuICT.core.layout.layout import Layout
from QuICT.qcda.mapping import *
from QuICT.tools.interface import OPENQASMInterface

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler import CouplingMap
from qiskit.transpiler.passes import  StochasticSwap

qubits_num = [5, 6, 7, 8, 9, 10]
gate_multiply = [5, 7, 9, 11, 13, 15]
layout_type = ["grid", "line"]

f = open("qiskit_mapping_benchmark_data.txt", 'w+')
for laytype in layout_type:
    f.write(f"{laytype}\n")
    for q_num in qubits_num:
        layout_fold = f"{laytype}{q_num}.layout"
        layout = Layout.load_file(
            os.path.dirname(os.path.abspath(__file__)) + 
            f"/../layout/{layout_fold}"
        )
        f.write(f"qubit_number: {q_num} \n")
        circuit_folder_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../Random_set/qiskit"
        )
        for gm in gate_multiply:
            f.write(f"gate size: {q_num * gm} \n")
            for i in range(1, 6):
                file_name = f"q{q_num}-g{gm * q_num}-{i}.qasm"
                cir = OPENQASMInterface.load_file(
                    circuit_folder_path + '/' + file_name
                ).circuit

                cir_map = MCTSMapping(layout).execute(cir)
                swap_gate_num = cir_map.count_gate_by_gatetype(GateType.swap)
                f.write(f"{i}, Quict swap gates number: {swap_gate_num} \n")
        
                with open(os.path.dirname(os.path.abspath(__file__)) + 
                f"/../layout/{layout_fold}") as file:
                    lines = file.readlines()
                lines_q = [line.rstrip() for line in lines]
                lines_mapping = []
                for str in lines_q[2:]:
                    a, b = str.split(' ')
                    lines_mapping.append([int(a), int(b)])
                
                cir_q = QuantumCircuit.from_qasm_file(circuit_folder_path + '/' + file_name)
                # print(cir_q)
                coupling_map = CouplingMap(couplinglist=lines_mapping)
                ss = StochasticSwap(coupling_map=coupling_map)
                pass_manager = PassManager(ss)
                stochastic_circ = pass_manager.run(cir_q)
                f.write(f"{i}, Qiskit swap gates number: {stochastic_circ.size()-cir_map.size()}\n")
        
f.close()