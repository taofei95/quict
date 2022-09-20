import os
from QuICT.tools.interface import OPENQASMInterface
from QuICT.qcda.optimization import *
from QuICT.qcda.optimization.symbolic_clifford_optimization import SymbolicCliffordOptimization
from QuICT.tools.interface.qasm_interface import OPENQASMInterface


from qiskit import QuantumCircuit, transpile


qubits_num = [5, 6, 7, 8, 9, 10]
gate_multiply = [5, 7, 9, 11, 13, 15]

f = open("qiskit_optimization_benchmark_data_2.txt", 'w+')
for q_num in qubits_num:
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

            cir_opt = CommutativeOptimization().execute(cir)

            f.write(f"{i}, Quict circuit size [ori:opt]=[{cir.size()}:{cir_opt.size()}] ; circuit depth [ori:opt]=[{cir.depth()}:{cir_opt.depth()}] \n")
            
            circ = QuantumCircuit.from_qasm_file(circuit_folder_path + '/' + file_name)
            circ_opt = transpile(circuits=circ, optimization_level=2)
            f.write(f"{i}, Qiskit circuit size [ori:opt]=[{circ.size()}:{circ_opt.size()}] ; circuit depth [ori:opt]=[{circ.depth()}:{circ_opt.depth()}] \n")
        
f.close()