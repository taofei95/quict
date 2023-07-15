from QuICT.algorithm.quantum_algorithm import QAE, OracleInfo

from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator


# f(x)=x1⊕x2...⊕xn
def example_oracle(n):
    def S_chi(n, controlled=False):
        # phase-flip on target
        cgate = CompositeGate()
        if controlled:
            for i in range(1, n + 1):
                CZ | cgate([0, i])
        else:
            for i in range(n):
                Z | cgate(i)
        return cgate

    def is_good_state(state_string):
        if len(state_string) == 0:
            return False
        else:
            return bool(int(state_string[0])) ^ is_good_state(state_string[1:])

    return OracleInfo(n=n, n_ancilla=0, S_chi=S_chi, is_good_state=is_good_state)


def test_canonical_QAE_run():
    pr_success = 0
    n_sample = 0
    for n in range(3, 5):
        eps = 0.1
        oracle = example_oracle(n)

        pr_function_good = 0
        for i in range(1 << n):
            if oracle.is_good_state(bin(i)[2:].rjust(n, "0")):
                pr_function_good += 1
        pr_function_good /= 1 << n

        n_local_sample = 10
        n_sample += n_local_sample
        for i in range(n_local_sample):
            pr_quantum_good = QAE(
                mode="canonical", eps=eps, simulator=StateVectorSimulator()
            ).run(oracle=oracle)
            print(f"{pr_quantum_good:.3f} from {pr_function_good:.3f}")
            if np.abs(pr_function_good - pr_quantum_good) < eps:
                pr_success += 1
    pr_success /= n_sample
    print(f"success rate {pr_success:.2f} with {n_sample:4} samples")
    assert pr_success > 8 / (np.pi ** 2)


def test_max_likely_QAE_run():
    pr_success = 0
    n_sample = 0
    for n in range(3, 5):
        eps = 0.1
        oracle = example_oracle(n)

        pr_function_good = 0
        for i in range(1 << n):
            if oracle.is_good_state(bin(i)[2:].rjust(n, "0")):
                pr_function_good += 1
        pr_function_good /= 1 << n

        n_local_sample = 10
        n_sample += n_local_sample
        for i in range(n_local_sample):
            pr_quantum_good = QAE(mode="max_likely", eps=eps).run(oracle=oracle)
            print(f"{pr_quantum_good:.3f} from {pr_function_good:.3f}")
            if np.abs(pr_function_good - pr_quantum_good) < eps:
                pr_success += 1
    pr_success /= n_sample
    print(f"success rate {pr_success:.2f} with {n_sample:4} samples")
    assert pr_success > 0.8  # a more rigid bound?


def test_fast_QAE_run():
    pr_success = 0
    n_sample = 0
    for n in range(3, 5):
        eps = 0.1
        oracle = example_oracle(n)

        pr_function_good = 0
        for i in range(1 << n):
            if oracle.is_good_state(bin(i)[2:].rjust(n, "0")):
                pr_function_good += 1
        pr_function_good /= 1 << n

        n_local_sample = 10
        n_sample += n_local_sample
        for i in range(n_local_sample):
            pr_quantum_good = QAE(mode="fast", eps=eps).run(oracle=oracle)
            print(f"{pr_quantum_good:.3f} from {pr_function_good:.3f}")
            if np.abs(pr_function_good - pr_quantum_good) < eps:
                pr_success += 1
    pr_success /= n_sample
    print(f"success rate {pr_success:.2f} with {n_sample:4} samples")
    from QuICT.algorithm.quantum_algorithm.amplitude_estimate import FQAE_DELTA
    assert pr_success > 1 - FQAE_DELTA
