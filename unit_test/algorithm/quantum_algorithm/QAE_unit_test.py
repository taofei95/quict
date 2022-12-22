import pytest

from QuICT.algorithm.quantum_algorithm import QAE, StatePreparationInfo, OracleInfo

from QuICT.core.gate import *
from QuICT.simulation.state_vector import ConstantStateVectorSimulator


def example_oracle(n):
    def S_chi(n, controlled=False):
        # phase-flip on target
        cgate = CompositeGate()
        if controlled:
            H | cgate(2)
            CCX | cgate([0, 1, 2])
            H | cgate(2)
        else:
            CZ | cgate([0, 1])
        return cgate

    def is_good_state(state_string):
        return state_string[:2] == "11"

    return OracleInfo(n=n, n_ancilla=0, S_chi=S_chi, is_good_state=is_good_state)


def test_canonical_QAE_run():
    n = 3
    eps = 0.1
    oracle = example_oracle(n)

    pr_function_good = 0
    for i in range(1 << n):
        if oracle.is_good_state(bin(i)[2:].rjust(n, "0")):
            pr_function_good += 1
    pr_function_good /= 1 << n

    pr_success = 0
    n_sample = 100
    for i in range(n_sample):
        pr_quantum_good = QAE(
            mode="canonical", eps=eps, simulator=ConstantStateVectorSimulator()
        ).run(oracle=oracle)
        print(f"{pr_quantum_good:.3f} from {pr_function_good:.3f}")
        if np.abs(pr_function_good - pr_quantum_good) < eps:
            pr_success += 1
    pr_success /= n_sample
    print(f"success rate {pr_success:.2f} with {n_sample:4} samples")
    assert pr_success > 8 / (np.pi ** 2)


def test_max_likely_QAE_run():
    n = 3
    eps = 0.05
    oracle = example_oracle(n)

    pr_function_good = 0
    for i in range(1 << n):
        if oracle.is_good_state(bin(i)[2:].rjust(n, "0")):
            pr_function_good += 1
    pr_function_good /= 1 << n

    pr_success = 0
    n_sample = 100
    for i in range(n_sample):
        pr_quantum_good = QAE(mode="max_likely", eps=eps).run(oracle=oracle)
        print(f"{pr_quantum_good:.3f} from {pr_function_good:.3f}")
        if np.abs(pr_function_good - pr_quantum_good) < eps:
            pr_success += 1
    pr_success /= n_sample
    print(f"success rate {pr_success:.2f} with {n_sample:4} samples")
    assert pr_success > 0.8  # a more rigid bound?


def test_fast_QAE_run():
    n = 3
    eps = 0.02
    oracle = example_oracle(n)

    pr_function_good = 0
    for i in range(1 << n):
        if oracle.is_good_state(bin(i)[2:].rjust(n, "0")):
            pr_function_good += 1
    pr_function_good /= 1 << n

    pr_success = 0
    n_sample = 100
    for i in range(n_sample):
        pr_quantum_good = QAE(mode="fast", eps=eps).run(oracle=oracle)
        print(f"{pr_quantum_good:.3f} from {pr_function_good:.3f}")
        if np.abs(pr_function_good - pr_quantum_good) < eps:
            pr_success += 1
    pr_success /= n_sample
    print(f"success rate {pr_success:.2f} with {n_sample:4} samples")
    from QuICT.algorithm.quantum_algorithm.amplitude_estimate import FQAE_DELTA
    assert pr_success > 1 - FQAE_DELTA
