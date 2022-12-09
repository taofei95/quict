import sys

from QuICT.tools.circuit_library import CircuitLib


def alg_circuit_genertor(alg: str, qubits: str, output_path: str):
    # Convert qubits from str into List[int]
    str_q_list = qubits.split("-")
    int_q_list = [int(sq) for sq in str_q_list]
    circuit_library = CircuitLib(output_type="file", output_path=output_path)

    circuit_library.get_algorithm_circuit(alg, max_size=int_q_list)
    # get_circuit(type, classify, width: list/slice?[list of qubits], max_size: int, max_depth: int [limitation the circuit's scale])


if __name__ == "__main__":
    alg_circuit_genertor(
        *sys.argv[1:]
    )
