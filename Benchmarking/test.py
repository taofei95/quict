





from Benchmarking import Benchmarking


cir_list = Benchmarking().get_circuit(type="random", classify="google", max_width=5, max_size=20, max_depth=20)
circuit = Benchmarking().filter_circuit(cir_list)
print(circuit)
