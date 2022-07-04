# Author  : Zhu Qinlin

"""
The (2n+2)-qubit circuit used in the Shor algorithm is designed \
by THOMAS HANER, MARTIN ROETTELER, and KRYSTA M. SVORE in \
"Factoring using 2n+2 qubits with Toffoli based modular multiplication"
"""

import logging
from math import pi
import numpy as np
from fractions import Fraction

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.arithmetic.hrs import *
from QuICT.algorithm import Algorithm
from .utility import *

from QuICT.simulation.cpu_simulator import CircuitSimulator


def construct_circuit(a: int, N: int, eps: float = 1 / 10):
    # phase estimation procedure
    n = int(np.ceil(np.log2(N + 1)))
    t = int(2 * n + 1 + np.ceil(np.log(2 + 1 / (2 * eps))))
    logging.info(f"\torder_finding circuit construction: circuit: n = {n} t = {t}")

    circuit = Circuit(2 * n + 1 + t)
    x_reg = list(range(n))  # n
    b_reg = list(range(n, 2 * n))  # n
    trickbits = list(range(2 * n, 2 * n + t))  # t
    indicator = [2 * n + t]  # 1
    X | circuit(x_reg[n - 1])

    for idx in trickbits:
        H | circuit(idx)
    for k in range(t):
        gate_pow = pow(a, 1 << (t - 1 - k), N)
        CHRSMulMod.execute(n, gate_pow, N) | circuit(
            x_reg + b_reg + indicator + [trickbits[k]]
        )
    for k in range(len(trickbits) // 2):
        Swap | circuit([trickbits[k], trickbits[len(trickbits) - 1 - k]])
    IQFT.build_gate(len(trickbits)) | circuit(trickbits)
    for k in range(len(trickbits) // 2):
        Swap | circuit([trickbits[k], trickbits[len(trickbits) - 1 - k]])
    for idx in b_reg + trickbits + indicator:
        Measure | circuit(idx)
    return circuit, trickbits[::-1]  # for int(circuit[trickbits]) convenience


def order_finding(a: int, N: int, eps: float = 1 / 10, simulator=CircuitSimulator()):
    circuit, trickbits = construct_circuit(a, N, eps)
    simulator.run(circuit)
    t = len(trickbits)

    # continued fraction procedure
    phi_ = int(circuit[trickbits]) / (1 << t)
    logging.info(f"\tphi~ (approximately s/r) in decimal form is {phi_}")
    r = Fraction(phi_).limit_denominator(N - 1).denominator
    logging.info(f"\tclose fraction form: {Fraction(phi_).limit_denominator(N - 1)}")
    return r


MAX_ROUND = 3


def reinforced_order_finding(
    a: int, N: int, eps: float = 1 / 10, simulator=CircuitSimulator()
):
    circuit, trickbits = construct_circuit(a, N, eps)
    t = len(trickbits)
    # continued fraction procedure (repetition)
    r_list = []
    i = 0
    while i < MAX_ROUND:
        i += 1
        simulator.run(circuit)
        phi_ = int(circuit[trickbits]) / (1 << t)
        logging.info(
            f"\tclose fraction form (repetition {i}): {Fraction(phi_).limit_denominator(N - 1)}"
        )
        r = Fraction(phi_).limit_denominator(N - 1).denominator
        if r != 0 and (a ** r) % N == 1:
            logging.info("\tsuccess!")
            r_list.append(r)
    if len(r_list) == 0:
        return 0
    else:
        return reduce(lambda x, y: (x * y) // gcd(x, y), r_list)