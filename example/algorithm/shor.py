from QuICT.algorithm.quantum_algorithm import ShorFactor

N = 35
sf = ShorFactor(mode="BEA_zip")
output = sf.run(N)
print(output)
