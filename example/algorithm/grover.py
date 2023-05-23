from QuICT.algorithm.quantum_algorithm import Grover
from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.core.gate.backend import MCTOneAux


def main_oracle(n, f):
    result_q = [n]
    cgate = CompositeGate()
    target_binary = bin(f[0])[2:].rjust(n, "0")
    with cgate:
        X & result_q[0]
        H & result_q[0]
        for i in range(n):
            if target_binary[i] == "0":
                X & i
    MCTOneAux().execute(n + 2) | cgate
    with cgate:
        for i in range(n):
            if target_binary[i] == "0":
                X & i
        H & result_q[0]
        X & result_q[0]
    return 2, cgate


n = 4
target = 0b0110
f = [target]
k, oracle = main_oracle(n, f)
grover = Grover(simulator=StateVectorSimulator())
result = grover.run(n, k, oracle)
print(result)
