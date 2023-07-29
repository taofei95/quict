from QuICT.core.virtual_machine.quantum_machine import OriginalKFC6130
from QuICT.benchmark import BenchmarkCircuitBuilder


def get_benchmark_circuit(N: int):
    width = OriginalKFC6130.qubit_number
    level = 1
    iset = OriginalKFC6130.instruction_set
    layout = OriginalKFC6130.layout

    cir_list = BenchmarkCircuitBuilder.get_benchmark_circuit(width, level, iset, layout, N)

    print(len(cir_list))


if __name__ == "__main__":
    get_benchmark_circuit(4)
