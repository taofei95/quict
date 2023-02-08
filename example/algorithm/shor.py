from QuICT.algorithm.quantum_algorithm import ShorFactor


N = 15
sf = ShorFactor(mode="BEA_zip")
output = sf.run(N, forced_quantum_approach=True)
print(output)
