import os
from QuICT.core.gate import CompositeGate
from QuICT.qcda.optimization import *
from QuICT.qcda.optimization.symbolic_clifford_optimization import SymbolicCliffordOptimization
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

# qubit_num = [5, 6, 7, 8, 9, 10]
# gate_multiply = [5, 7, 9, 11, 13, 15]
# instruction_set = ["Grover"]

# f = open("Grover_commutative_optimization_benchmark_data.txt", 'w+')
# for q_num in qubit_num:
#     f.write(f"qubit_number: {q_num} \n")
#     for iset in instruction_set:
#         f.write(f"{iset}\n")
#         circuit_folder_path = os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "../new",
#                 iset
#             )
#         for gm in gate_multiply:
#             f.write(f"gate size: {q_num * gm} \n")
#             for i in range(3, 16):
#                 if i % 2 == 1:
#                     file_name = f"grover_{i}.qasm"
#                     cir = OPENQASMInterface.load_file(
#                         circuit_folder_path + '/' + file_name
#                     ).circuit
#                     cir_depth = cir.depth()
#                     cir_opt = CommutativeOptimization().execute(cir)
#                     gate_num = cir_opt.size()
#                     cir_dep = cir_opt.depth()
#                     f.write(f"ori circuit depth: {cir_depth}, gate size: {gate_num}, opt circuit depth: {cir_dep} \n")
#                 else:
#                     pass
# f.close()


# qubit_num = [5, 6, 7, 8, 9, 10]
# gate_multiply = [5, 7, 9, 11, 13, 15]
# instruction_set = ["QFT"]

# f = open("QFT_commutative_optimization_benchmark_data.txt", 'w+')
# for q_num in qubit_num:
#     f.write(f"qubit_number: {q_num} \n")
#     for iset in instruction_set:
#         f.write(f"{iset}\n")
#         circuit_folder_path = os.path.join(
#                 os.path.dirname(os.path.abspath(__file__)),
#                 "../new",
#                 iset
#             )
#         for gm in gate_multiply:
#             f.write(f"gate size: {q_num * gm} \n")
#             for i in range(2, 16):
#                 file_name = f"qft_{i}.qasm"
#                 cir = OPENQASMInterface.load_file(
#                     circuit_folder_path + '/' + file_name
#                 ).circuit
#                 cir_depth = cir.depth()
#                 cir_opt = CommutativeOptimization().execute(cir)
#                 gate_num = cir_opt.size()
#                 cir_dep = cir_opt.depth()
#                 f.write(f"ori circuit depth: {cir_depth}, gate size: {gate_num}, opt circuit depth: {cir_dep} \n")
                
# f.close()

# qubit_num = [5, 6, 7, 8, 9, 10]
# gate_multiply = [5, 7, 9, 11, 13, 15]
# instruction_set = ["VQE"]
# name_list = ["HEA", "SPA"] #, "UCC"]

# f = open("VQE_commutative_optimization_benchmark_data.txt", 'w+')
# for q_num in qubit_num:
#     f.write(f"qubit_number: {q_num} \n")
#     for iset in instruction_set:
#         f.write(f"{iset}\n")
#         circuit_folder_path = os.path.join(
#             os.path.dirname(os.path.abspath(__file__)),
#             "../new",
#             iset
#             )
#         for gm in gate_multiply:
#             f.write(f"gate size: {q_num * gm} \n")
#             for name in name_list:
#                 for i in range(5, 9):
#                     file_name = f"{name}_{i}.qasm"
#                     cir = OPENQASMInterface.load_file(
#                         f"/home/wangrui/quict/wr_unit_test/benchmark/new/{iset}/{file_name}"
#                     ).circuit
#                     cir_depth = cir.depth()
#                     cir_opt = CommutativeOptimization().execute(cir)
#                     gate_num = cir_opt.size()
#                     cir_dep = cir_opt.depth()
#                     f.write(f"ori circuit depth: {cir_depth}, gate size: {gate_num}, opt circuit depth: {cir_dep} \n")
                
# f.close()

qubit_num = [5, 6, 7, 8, 9, 10]
gate_multiply = [5, 7, 9, 11, 13, 15]
instruction_set = ["Clifford"]
name_list = ["AG", "GD"]

f = open("Clifford_commutative_optimization_benchmark_data.txt", 'w+')
for q_num in qubit_num:
    f.write(f"qubit_number: {q_num} \n")
    for iset in instruction_set:
        f.write(f"{iset}\n")
        circuit_folder_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../new",
            iset
            )
        for gm in gate_multiply:
            f.write(f"gate size: {q_num * gm} \n")
            for name in name_list:
                for i in range(2, 15):
                    file_name = f"rand_cliff_{i}_{name}.qasm"
                    cir = OPENQASMInterface.load_file(
                        f"/home/wangrui/quict/wr_unit_test/benchmark/new/{iset}/{file_name}"
                    ).circuit
                    cir_depth = cir.depth()

                    gates = CompositeGate(gates=cir.gates)
                    SCO = SymbolicCliffordOptimization()
                    gates_opt = SCO.execute(gates)

                    gate_num = gates_opt.size()
                    cir_dep = gates_opt.depth()
                    f.write(f"ori circuit depth: {cir_depth}, gate size: {gate_num}, opt circuit depth: {cir_dep} \n")
                    # print(1)
f.close()
