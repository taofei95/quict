import pytest

from QuICT.algorithm.quantum_algorithm.amplitude_estimate import QAE
from QuICT.algorithm.quantum_algorithm.amplitude_estimate import StatePreparationInfo, OracleInfo

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import ConstantStateVectorSimulator
from QuICT.qcda.synthesis.mct import MCTOneAux

def example_oracle(n):
    def S_chi(n, controlled=False):
        # phase-flip on target
        cgate = CompositeGate()
        if controlled:
            H | cgate(2)
            CCX | cgate([0,1,2])
            H | cgate(2)
        else:
            CZ | cgate([0,1])
        return cgate

    def is_good_state(state_string):
        return state_string[:2]=='11'
    
    return OracleInfo(
        n=n,
        n_ancilla=0,
        S_chi=S_chi,
        is_good_state=is_good_state
    )

# def test_controlled_Q(): # check example_oracle
#     n = 3
#     oracle = example_oracle(n)
#     state_prep = StatePreparationInfo(n=n)
#     n_solution = len(list(filter(lambda x: oracle.is_good_state(bin(x)[2:].rjust(n,'0')),list(range(1<<n)))))
#     indices = list(range(1,1+n))
#     circ = Circuit(1+n+oracle.n_ancilla)
#     X | circ(0)
#     # state prepare
#     state_prep.A(n) | circ(indices)
#     # grover iter
#     N = 2 ** n
#     theta = 2 * np.arccos(np.sqrt(1 - n_solution / N))
#     T = int(round(np.arccos(np.sqrt(n_solution / N)) / theta))
#     print(T)
#     for i in range(T):
#         Q(n, True) | circ
#     amp = ConstantStateVectorSimulator().run(circ)
#     result = trace_prob(amp,indices)
#     pr_success = 0
#     for i in range(1<<n):
#         if is_good_state(bin(i)[2:].rjust(n,'0')):
#             pr_success += result[i]
#             print(f"{pr_success:.3f}")
#     assert pr_success > 0.9

# def test_good_state():
#     n = 4
#     circ = Circuit(n)
#     for i in range(n):
#         H | circ(i)
#     S_chi(n)[1] | circ
#     amp = ConstantStateVectorSimulator().run(circ)
#     amp = cp.asnumpy(amp)
#     pr = np.power(np.abs(amp), 2)
#     ps = np.real(np.log(amp) / (2j * np.pi))
#     for i in range(1<<n):
#         quantum_good = np.isclose(ps[i],0.5)
#         function_good = is_good_state(bin(i)[2:].rjust(n,'0'))
#         print(f"{ps[i]:.2f} for {bin(i)[2:].rjust(n,'0')}")
#         if quantum_good^function_good:
#             assert False

def test_canonical_QAE_run():
    n = 3
    eps = 0.1
    oracle = example_oracle(n)

    pr_function_good = 0
    for i in range(1<<n):
        if oracle.is_good_state(bin(i)[2:].rjust(n,'0')):
            pr_function_good += 1
    pr_function_good /= (1<<n)

    pr_success = 0
    n_sample = 100
    for i in range(n_sample):
        pr_quantum_good = QAE(mode="canonical",eps=eps,simulator=ConstantStateVectorSimulator()).run(oracle=oracle)
        print(f"{pr_quantum_good:.3f} from {pr_function_good:.3f}")
        if np.abs(pr_function_good-pr_quantum_good)<eps:
            pr_success += 1
    pr_success /= n_sample
    print(f"success rate {pr_success:.2f} with {n_sample:4} samples")
    assert pr_success > 8/(np.pi**2)

def test_max_likely_QAE_run():
    n = 3
    eps = 0.05
    oracle = example_oracle(n)

    pr_function_good = 0
    for i in range(1<<n):
        if oracle.is_good_state(bin(i)[2:].rjust(n,'0')):
            pr_function_good += 1
    pr_function_good /= (1<<n)

    pr_success = 0
    n_sample = 100
    for i in range(n_sample):
        pr_quantum_good = QAE(mode="max_likely",eps=eps).run(oracle=oracle)
        print(f"{pr_quantum_good:.3f} from {pr_function_good:.3f}")
        if np.abs(pr_function_good-pr_quantum_good)<eps:
            pr_success += 1
    pr_success /= n_sample
    print(f"success rate {pr_success:.2f} with {n_sample:4} samples")
    assert pr_success > 0.8 # a more rigid bound?

def test_fast_QAE_run():
    n = 3
    eps = 0.02
    oracle = example_oracle(n)

    pr_function_good = 0
    for i in range(1<<n):
        if oracle.is_good_state(bin(i)[2:].rjust(n,'0')):
            pr_function_good += 1
    pr_function_good /= (1<<n)

    pr_success = 0
    n_sample = 100
    for i in range(n_sample):
        pr_quantum_good = QAE(mode="fast",eps=eps).run(oracle=oracle)
        print(f"{pr_quantum_good:.3f} from {pr_function_good:.3f}")
        if np.abs(pr_function_good-pr_quantum_good)<eps:
            pr_success += 1
    pr_success /= n_sample
    print(f"success rate {pr_success:.2f} with {n_sample:4} samples")
    from QuICT.algorithm.quantum_algorithm.amplitude_estimate.FQAE import DELTA
    assert pr_success > 1 - DELTA
