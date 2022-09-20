import os
from QuICT.core import circuit
from QuICT.core.utils.gate_type import GateType

from QuICT.qcda.synthesis.gate_transform import GateTransform, InstructionSet
from QuICT.qcda.synthesis.gate_transform.special_set import USTCSet, IonQSet, IBMQSet
from QuICT.tools.interface.qasm_interface import OPENQASMInterface


qubit_num = [5, 6, 7, 8, 9, 10]
gate_multiply = [5, 7, 9, 11, 13, 15]
NomSet = InstructionSet(GateType.cx, [GateType.h, GateType.rz])
instruction_set = {
    "USTC": USTCSet,
    "IONQ": IonQSet,
    "IBMQ": IBMQSet,
    "Nom": NomSet
}

circuit_folder_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../new"
)

f = open("instruction_set_gate_transform_synthesis_benchmark_data.txt", 'w+')
for name, iset in instruction_set.items():
    f.write(f"instruction set: {name}\n")
    gt = GateTransform(iset)
    for q_num in qubit_num:
        f.write(f"qubits: {q_num}\n")
        for gm in gate_multiply:
            mean_size = 0
            mean_depth = 0
            for i in range(10):
                file_name = f"q{q_num}-g{q_num * gm}-{i}.qasm"
                cir = OPENQASMInterface.load_file(
                    circuit_folder_path + '/' + file_name
                    ).circuit

                cir_opt = gt.execute(cir)
                gate_num = cir_opt.size()
                cir_dep = cir_opt.depth()
                mean_size += gate_num
                mean_depth += cir_dep
                f.write(
                    f"{i}, ori circuit size: {cir.size()}; ori circuit depth: {cir.depth()};" +
                    f"opt circuit size:{gate_num},opt circuit depth:{cir_dep}"
                )
                f.write(f"mean opt circuit size: {mean_size}; mean opt circuit depth: {mean_depth}.")

f.close()
