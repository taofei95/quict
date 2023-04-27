import numpy as np

from QuICT.algorithm.quantum_algorithm.hhl import HHL
from QuICT.simulation.state_vector import StateVectorSimulator


def matrix(bits, cond_bits):
    n = 1 << bits
    eps = 1e-8
    while 1:
        m = np.complex128(np.random.rand(n, n))
        if np.abs(np.linalg.det(m)) > eps:
            ev = np.abs(np.linalg.eigvals(m))
            if np.max(ev) / np.min(ev) < (1 << cond_bits):
                return m


def vector(bits):
    n = 1 << bits
    eps = 1e-8
    while 1:
        v = np.complex128(np.random.rand(n))
        if abs(v.dot(v)) != eps:
            return v


def MSE(x, y):
    n = len(x)
    res0 = np.dot(x - y, x - y) / n
    res1 = np.dot(x + y, x + y) / n
    return min(res0, res1)


A = matrix(1, 6)
b = vector(1)
A = np.array([[1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j],
              [0.0 + 0j, 2.0 + 0j, 0.0 + 0j, 0.0 + 0j],
              [0.0 + 0j, 0.0 + 0j, 3.0 + 0j, 0.0 + 0j],
              [0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 4.0 + 0j]])
b = np.array([1.0 + 0j, -2.0 + 0j, 1.0 + 0j, 1.0 + 0j])
print("start")
slt = np.linalg.solve(A, b)
slt /= np.linalg.norm(slt)

hhl_a = HHL(StateVectorSimulator()).run(
            A, b, phase_qubits=6, measure=False
        )
# hhl_a /= np.linalg.norm(hhl_a)

# time = 0
# hhl = None
# HHL_m = HHL(StateVectorSimulator(device="GPU"))
# while(hhl is None):
#     hhl = HHL_m.run(
#             A, b, phase_qubits=9
#         )
#     time += 1

print(f"classical solution  = {slt}\n"
    + f"hhl without measure = {hhl_a}\n"
    # + f"                MSE = {MSE(slt, hhl_a)}\n"
    # + f"hhl with measure    = {hhl}\n"
    # + f"                MSE = {MSE(slt, hhl)}\n"
    # + f"       success rate = {1.0 / time}"
    )
