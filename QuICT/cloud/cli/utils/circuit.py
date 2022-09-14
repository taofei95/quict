import os
import shutil

from QuICT.core import Circuit
from QuICT.core.gate import GateType
from QuICT.lib.circuitlib import CircuitLib
from .helper_function import path_check, qasm_validation


default_customed_circuit_folder = os.path.join(
    os.path.dirname(__file__),
    "..",
    "circuit"
)


@path_check
def get_random_circuit(
    qubits: list,
    size: list,
    random_param: bool,
    instruction_set: str = "random",
    output_path: str = '.'
):
    google_set = [GateType.sx, GateType.sy, GateType.sw, GateType.rx, GateType.ry, GateType.fsim]
    ibmq_set = [GateType.rz, GateType.sx, GateType.x, GateType.cx]
    ionq_set = [GateType.rx, GateType.ry, GateType.rz, GateType.rxx]
    ustc_set = [GateType.rx, GateType.ry, GateType.rz, GateType.h, GateType.x, GateType.cx]
    based_set = [
        GateType.rx, GateType.ry, GateType.rz,
        GateType.cx, GateType.cy, GateType.crz,
        GateType.ch, GateType.cz, GateType.rxx,
        GateType.ryy, GateType.rzz, GateType.fsim
    ]
    iset_mapping = {
        "Google": google_set,
        "USTC": ustc_set,
        "IBMQ": ibmq_set,
        "IONQ": ionq_set,
        "random": based_set
    }

    gate_set = iset_mapping[instruction_set]
    prob = [3 / (4 * (len(gate_set) - 1))] * (len(gate_set) - 1) + [1 / 4] if instruction_set != "random" else \
        [1 / len(gate_set)] * len(gate_set)

    for q in qubits:
        for s in size:
            cir_file_name = f"{instruction_set}_{q}_{s}.qasm"
            cir = Circuit(q)
            cir.random_append(s, gate_set, random_param, prob)

            with open(f"{output_path}/{cir_file_name}", "w+") as f:
                f.write(cir.qasm())


@path_check
def get_algorithm_circuit(alg: str, qubits: list, output_path: str = "."):
    cir_list = CircuitLib().get_circuit("algorithm", alg, qubits)
    for cir in cir_list:
        file_name = f"{alg}_{cir.width()}.qasm"
        with open(f"{output_path}/{file_name}", "w+") as f:
            f.write(cir.qasm())


def store_quantum_circuit(name: str, file: str):
    get_folder_name = os.listdir(default_customed_circuit_folder)
    if not name.endswith(".qasm"):
        name += ".qasm"

    if name in get_folder_name:
        raise KeyError("Repeat circuits name.")

    # qasm file validation
    qasm_validation(file)
    shutil.copy(file, f"{default_customed_circuit_folder}/{name}")


def delete_quantum_circuit(name: str):
    get_folder_name = os.listdir(default_customed_circuit_folder)
    if not name.endswith(".qasm"):
        name += ".qasm"

    if name not in get_folder_name:
        raise KeyError("No target name in circuit list.")

    os.remove(f"{default_customed_circuit_folder}/{name}")


def list_quantum_circuit():
    print(os.listdir(default_customed_circuit_folder))
