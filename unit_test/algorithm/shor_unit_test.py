from math import gcd
import random
from QuICT.algorithm.quantum_algorithm import ShorFactor
from QuICT.algorithm.quantum_algorithm.shor.BEA_zip import (
    order_finding as BEA_order_finding,
)
from QuICT.algorithm.quantum_algorithm.shor.BEA_zip import (
    construct_circuit as BEA_construct_circuit,
)
from QuICT.algorithm.quantum_algorithm.shor.HRS_zip import (
    order_finding as HRS_order_finding,
)
from QuICT.algorithm.quantum_algorithm.shor.HRS_zip import (
    construct_circuit as HRS_construct_circuit,
)
import pytest

from QuICT.simulation.state_vector import StateVectorSimulator


simulator = StateVectorSimulator(device="CPU")
threthold_rate = 0.3
number_list = [
    9,
    15,
]
n_repeat = 5
order_finding_test_modes = {"BEA": BEA_order_finding, "HRS": HRS_order_finding}
order_finding_circuit_test_modes = {
    "BEA": BEA_construct_circuit,
    "HRS": HRS_construct_circuit,
}
run_test_modes = {"BEA_zip"}
circuit_test_modes = {"BEA_zip", "BEA"}


def test_OrderFinding():
    for mode in order_finding_test_modes.keys():
        failure = 0
        for N in number_list:
            p = random.choice(
                list(filter(lambda x: gcd(x, N) == 1 and x != 1, list(range(N))))
            )
            for _ in range(n_repeat):
                print(f"testing ({p:2},{N:2})...", end="")
                a = order_finding_test_modes[mode](a=p, N=N, eps=1)
                print(f"{'T' if (p**a)%N==1 and a!=0 else 'F'}: {p}**{a}==1 mod {N}")
                if a == 0 or (p ** a) % N != 1:
                    failure += 1
    print(f"success rate: {1-failure/len(number_list)/n_repeat:.3f}")
    if 1 - failure / len(number_list) / n_repeat < threthold_rate:
        assert False


def test_OrderFinding_circuit():
    from fractions import Fraction

    for mode in order_finding_circuit_test_modes.keys():
        failure = 0
        for N in number_list:
            p = random.choice(
                list(filter(lambda x: gcd(x, N) == 1 and x != 1, list(range(N))))
            )
            print(f"testing ({p:2},{N:2})...", end="")
            for _ in range(n_repeat):
                circ, indices = order_finding_circuit_test_modes[mode](a=p, N=N, eps=1)
                trickbit = indices[0]
                indices = indices[1:]
                simulator.run(circ)
                phi = eval(
                    "0b"
                    + "".join(
                        [str(circ[trickbit].historical_measured[idx]) for idx in indices]
                    )
                ) / (1 << len(indices))
                a = Fraction(phi).limit_denominator(N - 1).denominator
                print(f"{'T' if (p**a)%N==1 and a!=0 else 'F'}: {p}**{a}==1 mod {N}")
                if a == 0 or (p ** a) % N != 1:
                    failure += 1
    print(f"success rate: {1-failure/len(number_list)/n_repeat:.3f}")
    if 1 - failure / len(number_list) / n_repeat < threthold_rate:
        assert False


def test_ShorFactor_run():
    for mode in run_test_modes:
        print(f"mode: {mode}")
        failure = 0
        for number in number_list:
            print("-------------------FACTORING %d-------------------------" % number)
            for _ in range(n_repeat):
                a = ShorFactor(mode=mode, eps=1, max_rd=1).run(N=number)
                if a == 0 or number % a != 0:
                    failure += 1
    print(f"success rate: {1-failure/len(number_list)/n_repeat:.3f}")
    if 1 - failure / len(number_list) / n_repeat < threthold_rate:
        assert False


def test_ShorFactor_circuit():
    for mode in circuit_test_modes:
        print(f"mode: {mode}")
        failure = 0
        for number in number_list:
            if number % 2 == 0 or number == 9:
                continue
            print("-------------------FACTORING %d-------------------------" % number)
            for _ in range(n_repeat):
                circuit, indices = ShorFactor(mode=mode, eps=1, max_rd=1).circuit(N=number)
                a = ShorFactor(mode=mode, simulator=simulator).run(
                    N=number, circuit=circuit, indices=indices
                )
                if a == 0 or number % a != 0:
                    failure += 1
    print(f"success rate: {1-failure/len(number_list)/n_repeat:.3f}")
    if 1 - failure / len(number_list) / n_repeat < threthold_rate:
        assert False
