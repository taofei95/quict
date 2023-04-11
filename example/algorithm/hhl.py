import numpy as np
from scipy import stats, sparse

from QuICT.algorithm.quantum_algorithm.hhl import HHL

from QuICT.simulation.state_vector import StateVectorSimulator


def random_matrix(size):
    rvs = stats.norm().rvs
    while(1):
        X = sparse.random(
            size, size, density=1, data_rvs=rvs,
            dtype=np.complex128)
        A = X.todense()
        v = np.linalg.eigvals(A)
        A = np.round(A, 3)
        if np.linalg.det(A) != 0 and np.log2(max(abs(v)) / min(abs(v))) < 6:
            return np.array(A)


def random_vector(size):
    return np.complex128(np.round(np.random.rand(size), 3) - np.full(size, 0.5))


n = 2 ** 1
A = random_matrix(n)
b = random_vector(n)

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
