import numpy as np

from QuICT.algorithm.quantum_algorithm.hhl import HHL
from QuICT.simulation.state_vector import StateVectorSimulator


def MSE(x, y):
    n = len(x)
    res0 = np.linalg.norm(x + y) / n
    res1 = np.linalg.norm(x - y) / n
    return min(res0, res1)


A = np.array([[1.0 + 0j, 2.0 + 0j],
              [3.0 + 0j, 2.0 + 0j]])
b = np.array([1.0 + 0j, -2.0 + 0j])

slt = np.linalg.solve(A, b)
slt /= np.linalg.norm(slt)

hhl_a = HHL(StateVectorSimulator(device="GPU")).run(
            A, b, phase_qubits=6, measure=False
        )
hhl_a /= np.linalg.norm(hhl_a)

time = 0
hhl = None
HHL_m = HHL(StateVectorSimulator(device="GPU"))
while(hhl is None):
    hhl = HHL_m.run(
            A, b, phase_qubits=6
        )
    time += 1

print(f"classical solution  = {slt}\n"
    + f"hhl without measure = {hhl_a}\n"
    + f"                MSE = {MSE(slt, hhl_a)}\n"
    + f"hhl with measure    = {hhl}\n"
    + f"                MSE = {MSE(slt, hhl)}\n"
    + f"       success rate = {1.0 / time}"
    )
