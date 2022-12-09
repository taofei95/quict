import sys

from QuICT.core import Circuit
from QuICT.tools.circuit_library import CircuitLib


def random_circuit_generator(
    classify: str,
    qubits: str,
    max_size: int,
    max_depth: int,
    output_path: str
):
    # Convert qubits and size from str into List[int]
    str_q_list = qubits.split("-")
    int_q_list = [int(sq) for sq in str_q_list]
    circuit_library = CircuitLib(output_type="file", output_path=output_path)

    circuit_library.get_random_circuit(classify, int_q_list, max_size, max_depth)


if __name__ == "__main__":
    random_circuit_generator(
        *sys.argv[1:]
    )
