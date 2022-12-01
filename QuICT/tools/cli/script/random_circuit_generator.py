import sys

from QuICT.core import Circuit
from QuICT.core.gate import GateType


def random_circuit_generator(
    qubits: str,
    size: str,
    random_param: bool,
    instruction_set: str,
    output_path: str
):
    google_set = [GateType.sx, GateType.sy, GateType.sw, GateType.rx, GateType.ry, GateType.fsim]
    ibmq_set = [GateType.rz, GateType.sx, GateType.x, GateType.cx]
    ionq_set = [GateType.rx, GateType.ry, GateType.rz, GateType.rxx]
    ustc_set = [GateType.rx, GateType.ry, GateType.rz, GateType.h, GateType.x, GateType.cx]

    iset_mapping = {
        "Google": google_set,
        "USTC": ustc_set,
        "IBMQ": ibmq_set,
        "IONQ": ionq_set
    }

    gate_set = iset_mapping[instruction_set] if instruction_set != "random" else None
    prob = [3 / (4 * (len(gate_set) - 1))] * (len(gate_set) - 1) + [1 / 4] if instruction_set != "random" else \
        None

    # Convert qubits and size from str into List[int]
    str_q_list = qubits.split("-")
    int_q_list = [int(sq) for sq in str_q_list]
    str_s_list = size.split("-")
    int_s_list = [int(sq) for sq in str_s_list]
    for q in int_q_list:
        for s in int_s_list:
            cir_file_name = f"{instruction_set}_{q}_{s}.qasm"
            cir = Circuit(q)
            cir.random_append(s, gate_set, random_param, prob)

            with open(f"{output_path}/{cir_file_name}", "w+") as f:
                f.write(cir.qasm())


if __name__ == "__main__":
    random_circuit_generator(
        *sys.argv[1:]
    )
