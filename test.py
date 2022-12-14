from QuICT.tools.circuit_library.get_benchmark_circuit import BenchmarkCircuitBuilder


cir = BenchmarkCircuitBuilder.mediate_measure_circuit_build(5, 5)
print(cir)