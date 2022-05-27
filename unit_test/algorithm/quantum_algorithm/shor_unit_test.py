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
import logging

from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator
from QuICT.simulation.cpu_simulator.cpu import CircuitSimulator

simulator = CircuitSimulator()
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
    22,
    24,
    25,
    26,
    27,
]
order_finding_test_modes = {"BEA": BEA_order_finding, "HRS": HRS_order_finding}
order_finding_circuit_test_modes = {
    "BEA": BEA_construct_circuit,
    "HRS": HRS_construct_circuit,
}
run_test_modes = {"BEA_zip", "HRS_zip"}
circuit_test_modes = {"BEA_zip", "HRS_zip", "BEA", "HRS"}

# def test_OrderFinding():
#     for mode in order_finding_test_modes.keys():
#         failure = 0
#         for N in number_list:
#             p = random.choice(list(filter(lambda x:gcd(x,N)==1,list(range(N)))))
#             print(f"testing ({p:2},{N:2})...",end="")
#             a = order_finding_test_modes[mode](p,N,simulator=simulator)
#             print(f"{'T' if (p**a)%N==1 else 'F'}: {p}**{a}==1 mod {N}")
#             if a==0 or (p**a)%N!=1:
#                 failure += 1
#     print(f"success rate: {1-failure/len(number_list):.3f}")

# def test_OrderFinding_circuit():
#     from fractions import Fraction
#     for mode in order_finding_circuit_test_modes.keys():
#         failure = 0
#         for N in number_list:
#             p = random.choice(list(filter(lambda x:gcd(x,N)==1,list(range(N)))))
#             print(f"testing ({p:2},{N:2})...",end="")
#             circ, indices = order_finding_circuit_test_modes[mode](p,N)
#             simulator.run(circ)
#             phi = eval("0b"+"".join([str(trig.measured[0]) for trig in indices]))/(1<<len(indices))
#             a = Fraction(phi).limit_denominator(N - 1).denominator
#             print(f"{'T' if (p**a)%N==1 else 'F'}: {p}**{a}==1 mod {N}")
#             if a==0 or (p**a)%N!=1:
#                 failure += 1
#     print(f"success rate: {1-failure/len(number_list):.3f}")

# def test_ShorFactor_run():
#     for mode in run_test_modes:
#         print(f"mode: {mode}")
#         failure = 0
#         for number in number_list:
#             print('-------------------FACTORING %d-------------------------' % number)
#             a = ShorFactor(mode=mode,N=number).run(simulator=simulator)
#             if a == 0 or number % a != 0:
#                 failure += 1
#         print(f"success rate: {1-failure/len(number_list):.3f}")


def test_ShorFactor_circuit():
    for mode in circuit_test_modes:
        print(f"mode: {mode}")
        failure = 0
        for number in number_list:
            print("-------------------FACTORING %d-------------------------" % number)
            circuit, indices = ShorFactor(mode=mode, N=number).circuit()
            a = ShorFactor(mode=mode, N=number).run(
                circuit=circuit, indices=indices, simulator=simulator
            )
            if a == 0 or number % a != 0:
                failure += 1
        print(f"success rate: {1-failure/len(number_list):.3f}")


# logging.root.setLevel(logging.INFO)
# test_OrderFinding_circuit()
