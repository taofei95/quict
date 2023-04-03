import numpy as np

from QuICT.algorithm.quantum_algorithm.hhl import HHL

from QuICT.simulation.state_vector import StateVectorSimulator

from example.algorithm.random_sparse import RS

n = 2 ** 5
A = RS(n, 1).matrix()
b = np.ones(n, dtype=np.complex128)

slt = np.linalg.solve(A, b)
slt /= np.linalg.norm(slt)
hhl_u = HHL(StateVectorSimulator(device="GPU")).run(
        matrix=A,
        vector=b)
if hhl_u is not None:
        hhl_u /= np.linalg.norm(hhl_u)
        print(f"solution     = {slt.real}\n" +
            f"hhl(unitary) = {hhl_u.real}")
else: 
        print("Failed.")
