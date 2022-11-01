import sys

from QuICT.lib.circuitlib import CircuitLib


def alg_circuit_genertor(alg: str, qubits: str, output_path: str):
    # Convert qubits from str into List[int]
    str_q_list = qubits.split("-")
    int_q_list = [int(sq) for sq in str_q_list]

    cir_list = CircuitLib().get_circuit("algorithm", alg, int_q_list)
    for cir in cir_list:
        file_name = f"{alg}_{cir.width()}.qasm"
        with open(f"{output_path}/{file_name}", "w+") as f:
            f.write(cir.qasm())


if __name__ == "__main__":
    print(sys.argv[1:])
    alg_circuit_genertor(
        *sys.argv[1:]
    )
