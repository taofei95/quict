from math import gcd
import random
from QuICT.algorithm.quantum_algorithm import ShorFactor

from QuICT.simulation.state_vector import StateVectorSimulator


simulator = StateVectorSimulator(device="GPU")
threthold_rate = 0.3

def _test_OrderFinding(mode, number_list=None):
    failure = 0
    if number_list is None:
        assert False
    for N,p in number_list:
        print(f"testing ({p:2},{N:2})...", end="")
        a = ShorFactor._RUN_METHOD_OF_MODE[mode](p, N, simulator=simulator)
        print(f"{'T' if (p**a)%N==1 and a!=0 else 'F'}: {p}**{a}==1 mod {N}")
        if a == 0 or (p ** a) % N != 1:
            failure += 1
    print(f"success rate: {1-failure/len(number_list):.3f}")

def _test_OrderFinding_circuit(mode, number_list):
    from fractions import Fraction
    failure = 0
    for N,p in number_list:
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


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str)
parser.add_argument('--number_list', type=str)
parser.add_argument('--circuit', type=bool, default=False)
args = parser.parse_args()
if not args.circuit:
    _test_OrderFinding(args.mode, number_list=eval(args.number_list))    
else:
    _test_OrderFinding_circuit(args.mode, number_list=eval(args.number_list))