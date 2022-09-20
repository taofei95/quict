import os
from QuICT.qcda.optimization import *
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
from QuICT.qcda.optimization.template_optimization.templates import (template_nct_2a_2,
                                                                     template_nct_4a_3,
                                                                     template_nct_5a_3,
                                                                     template_nct_6a_1,
                                                                     template_nct_9c_5,
                                                                     template_nct_9d_4)
from QuICT.qcda.optimization.template_optimization import TemplateOptimization

# qubit_num = [5, 6, 7, 8, 9, 10]
# gate_multiply = [5, 7, 9, 11, 13, 15]
# instruction_set = ["Grover"]

# f = open("Grover_template_optimization_benchmark_data.txt", 'w+')
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
#                         f"/home/wangrui/quict/wr_unit_test/benchmark/new/{iset}/{file_name}"
#                     ).circuit
#                     cir_depth = cir.depth()
                    
#                     templates = [template_nct_2a_2(),
#                                     template_nct_4a_3(),
#                                     template_nct_5a_3(),
#                                     template_nct_6a_1(),
#                                     template_nct_9c_5(),
#                                     template_nct_9d_4()]
#                     cir_opt = TemplateOptimization(templates).execute(cir)
                    
#                     gate_num = cir_opt.size()
#                     cir_dep = cir_opt.depth()
#                     f.write(f"{i}, ori circuit depth: {cir_depth}, gate size: {gate_num}, opt circuit depth: {cir_dep} \n")
#                 else:
#                     pass
# f.close()

qubit_num = [5, 6, 7, 8, 9, 10]
gate_multiply = [5, 7, 9, 11, 13, 15]
instruction_set = ["QFT"]

f = open("QFT_template_optimization_benchmark_data.txt", 'w+')
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
            for i in range(2, 16):
                file_name = f"qft_{i}.qasm"
                cir = OPENQASMInterface.load_file(
                    f"/home/wangrui/quict/wr_unit_test/benchmark/new/{iset}/{file_name}"
                ).circuit
                cir_depth = cir.depth()
                
                templates = [template_nct_2a_2(),
                            template_nct_4a_3(),
                            template_nct_5a_3(),
                            template_nct_6a_1(),
                            template_nct_9c_5(),
                            template_nct_9d_4()
                            ]
                cir_opt = TemplateOptimization(templates).execute(cir)
                
                gate_num = cir_opt.size()
                cir_dep = cir_opt.depth()
                f.write(f"{i}, ori circuit depth: {cir_depth}, gate size: {gate_num}, opt circuit depth: {cir_dep} \n")
f.close()