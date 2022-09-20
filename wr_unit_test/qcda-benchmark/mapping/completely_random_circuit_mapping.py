import os
from QuICT.core import Layout
from QuICT.core.gate import GateType
from QuICT.tools.interface import OPENQASMInterface
from QuICT.qcda.mapping import *

qubits_num = [5, 6, 7, 8, 9, 10]
gate_multiply = [5, 7, 9, 11, 13, 15]
layout_type = ["line", "grid"]
f = open("completely_circuit_mapping_benchmark_data.txt", 'w+')
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
            "../Random_set/new",
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
                f.write(f"{i}, swap gates number: {swap_gate_num} \n")

f.close()
