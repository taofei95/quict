from QuICT.tools.circuit_library import CircuitLib


def get_circuit_from_CLIB():
    cir_lib = CircuitLib()

    # Circuit Library's size
    print(cir_lib.size)

    # Get Circuit through circuit library
    cirs = cir_lib.get_circuit("algorithm", "qft", 10)
    print(len(cirs))
    print(cirs[-1].name)

    cirs = cir_lib.get_algorithm_circuit("grover", [3, 5, 7], max_depth=20)
    print(len(cirs))
    print(cirs[-1].name)

    cirs = cir_lib.get_random_circuit("diag", [3, 5, 7], 40, 20)
    print(len(cirs))
    print(cirs[-1].name)

    cirs = cir_lib.get_benchmark_circuit("highly_entangled", [3, 5], 20, 15)
    print(len(cirs))
    print(cirs[-1].name)

    # Get Circuit's QASM from Circuit Library
    cir_lib_qasm = CircuitLib(output_type="qasm")
    qasms = cir_lib_qasm.get_circuit("algorithm", "maxcut", 10)
    print(len(qasms))
    print(qasms[-1])

    qasms = cir_lib_qasm.get_random_circuit("single_bit", 4, 15, 15)
    print(len(qasms))
    print(qasms[-1])

    # Save to the given folder
    cir_lib_file = CircuitLib(output_type="file", output_path="./temp_list")
    cir_lib_file.get_circuit("template", "template", 3, 6, 5)


if __name__ == "__main__":
    get_circuit_from_CLIB()
