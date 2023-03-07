import pytest
import math

from QuICT.algorithm.quantum_algorithm import QPE

from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator

def test_QPE_0():
    for proportion in [0.25, 0.5, 0.75]:
        n = 3
        m = 2
        workbits = list(range(m))
        trickbits = list(range(m,m+n))
        constructor = lambda power: CRz(power*proportion*4*math.pi)
        state_prep = CompositeGate()
        with state_prep:
            X | state_prep(0)
        proportion_esti = QPE(workbits, trickbits, state_prep, constructor, StateVectorSimulator(device="GPU")).run()
        print(f"phase/2π: {proportion_esti}")
        assert math.isclose(proportion, proportion_esti)


# f(x)=x1⊕x2...⊕xn
def _example_grover_operator(n, power):
    from QuICT.core.gate.backend import MCTOneAux
    cgate = CompositeGate()
    # S_chi
    for i in range(1, n + 1):
        CZ | cgate([0, i])
    # A_dagger
    for i in range(1, n + 1):
        CH | cgate([0, i])
    # S_0
    indices = list(range(1, 1 + n))
    for i in indices:
        CX | cgate([0, i])
    CH | cgate([0, indices[n - 1]])
    MCTOneAux().execute(n + 2) | cgate
    CH | cgate([0, indices[n - 1]])
    for i in indices:
        CX | cgate([0, i])
    # A
    for i in range(1, n + 1):
        CH | cgate([0, i])
    
    # repeat
    cgate_final = CompositeGate()
    for _ in range(power):
        cgate | cgate_final
    return cgate_final

def test_QPE_1():
    for m in [3,4,5]:
        n = 4
        workbits = list(range(m))
        trickbits = list(range(m,m+n))
        constructor = lambda power: _example_grover_operator(m-1, power)
        state_prep = CompositeGate()
        with state_prep:
            for i in workbits:
                H | state_prep(i)
        proportion_esti = QPE(workbits, trickbits, state_prep, constructor, StateVectorSimulator(device="GPU")).run()
        amp = 1 - np.sin(np.pi * proportion_esti) ** 2
        print(f"amplitude: {amp:.3f}")
        assert abs(0.5 - amp) < 1/2**n
