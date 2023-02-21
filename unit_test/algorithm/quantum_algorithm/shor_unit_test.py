from math import gcd
import random
from QuICT.algorithm.quantum_algorithm import ShorFactor
import sys

from QuICT.simulation.state_vector import StateVectorSimulator


simulator = StateVectorSimulator(device="GPU")
threthold_rate = 0.3
number_list = [
    6,
    8,
    9,
    10,
    12,
    14,
    15,
    16,
    18,
    20,
    21,
    # 22,
    # 24,
    # 25,
    # 26,
    # 27,
]
run_test_modes = {"BEA_zip", "HRS_zip"}
circuit_test_modes = {"BEA_zip", "HRS_zip", "BEA", "HRS"}


def _test_OrderFinding(mode, custom_number_list=None):
    failure = 0
    if custom_number_list is None:
        custom_number_list = number_list
    for N in custom_number_list:
        p = random.choice(
            list(filter(lambda x: gcd(x, N) == 1 and x != 1, list(range(N))))
        )
        print(f"testing ({p:2},{N:2})...", end="")
        a = ShorFactor._RUN_METHOD_OF_MODE[mode](p, N, simulator=simulator)
        print(f"{'T' if (p**a)%N==1 and a!=0 else 'F'}: {p}**{a}==1 mod {N}")
        if a == 0 or (p ** a) % N != 1:
            failure += 1
    print(f"success rate: {1-failure/len(custom_number_list):.3f}")
    if 1 - failure / len(custom_number_list) < threthold_rate:
        assert False

def test_OrderFinding():
    for mode in ShorFactor._ALLOWED_MODES:
        _test_OrderFinding(mode)

def _test_OrderFinding_circuit(mode):
    from fractions import Fraction
    failure = 0
    for N in number_list:
        p = random.choice(
            list(filter(lambda x: gcd(x, N) == 1 and x != 1, list(range(N))))
        )
        print(f"testing ({p:2},{N:2})...", end="")
        circ, indices = ShorFactor._CIRCUIT_METHOD_OF_MODE[mode](p, N)
        simulator.run(circ)
        trickbit = indices[0]
        indices = indices[1:]
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
    print(f"success rate: {1-failure/len(number_list):.3f}")
    if 1 - failure / len(number_list) < threthold_rate:
        assert False

def test_OrderFinding_circuit():
    for mode in ShorFactor._ALLOWED_MODES:
        _test_OrderFinding_circuit(mode)

def test_ShorFactor_run():
    for mode in run_test_modes:
        print(f"mode: {mode}")
        failure = 0
        for number in number_list:
            print("-------------------FACTORING %d-------------------------" % number)
            a = ShorFactor(mode=mode).run(N=number)
            if a == 0 or number % a != 0:
                failure += 1
        print(f"success rate: {1-failure/len(number_list):.3f}")
        if 1 - failure / len(number_list) < threthold_rate:
            assert False


def test_ShorFactor_circuit():
    for mode in circuit_test_modes:
        # if mode in {"BEA_zip", "HRS_zip"}:
        #     raise AssertionError("clear circuit state e.g. hitorical_measured to run this mode")
        print(f"mode: {mode}")
        failure = 0
        for number in number_list:
            if number % 2 == 0 or number == 25 or number == 27:
                continue
            print("-------------------FACTORING %d-------------------------" % number)
            circuit, indices = ShorFactor(mode=mode).circuit(N=number)
            a = ShorFactor(mode=mode, simulator=simulator).run(
                N=number, circuit=circuit, indices=indices
            )
            if a == 0 or number % a != 0:
                failure += 1
        print(f"success rate: {1-failure/len(number_list):.3f}")
        if 1 - failure / len(number_list) < threthold_rate:
            assert False
