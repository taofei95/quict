# Author  : Peng Sirui

'''
The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
St´ephane Beauregard in "Circuit for Shor’s algorithm using 2n+3 qubits"\
'''

import logging
from math import pi, gcd
import numpy as np
from fractions import Fraction
from functools import reduce

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.arithmetic.bea import *
from QuICT.algorithm import Algorithm
from .utility import *

from QuICT.simulation.cpu_simulator import CircuitSimulator
from QuICT.simulation import Simulator


def order_finding(a: int, N: int, eps: float = 1 / 10, simulator: Simulator = CircuitSimulator()):
    """
    The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
    Stephane Beauregard in "Circuit for Shor's algorithm using 2n+3 qubits"

    Quantum algorithm to compute the order of a (mod N), when gcd(a,N)=1.
    """
    # phase estimation procedure
    n = int(np.ceil(np.log2(N + 1)))
    t = int(2 * n + 1 + np.ceil(np.log(2 + 1 / (2 * eps))))
    logging.info(f'\torder_finding begin: circuit: n = {n} t = {t}')
    trickbit_store = [0] * t

    b_reg = [i for i in range(n + 1)]
    x_reg = [i for i in range(n + 1, 2 * n + 1)]
    trickbit = [2 * n + 1]
    qreg_low = [2 * n + 2]
    # subcircuit: init \ket{1}\ket{0}
    circuit = Circuit(2 * n + 3)
    X | circuit(x_reg[n - 1])
    amp = simulator.run(circuit)

    for k in range(t):
        # subcircuit CUa
        circuit = Circuit(2 * n + 3)
        H | circuit(trickbit)
        gate_pow = pow(a, 1 << (t - 1 - k), N)
        BEACUa.execute(n, gate_pow, N) | circuit
        amp = simulator.run(circuit, use_previous=True)
        # subcircuit: semi-classical QFT
        circuit = Circuit(2 * n + 3)
        for i in range(k):
            if trickbit_store[i]:
                Rz(-pi / (1 << (k - i))) | circuit(trickbit)
        H | circuit(trickbit)
        amp = simulator.run(circuit, use_previous=True)
        circuit = Circuit(2 * n + 3)
        for idx in (b_reg + trickbit + qreg_low): Measure | circuit(idx)
        amp = simulator.run(circuit, use_previous=True)
        # subcircuit: measure & reset trickbit
        assert int(circuit[qreg_low]) == 0 and int(circuit[b_reg]) == 0
        logging.info(f'\tthe {k}th trickbit measured to be {int(circuit[trickbit])}')
        trickbit_store[k] = int(circuit[trickbit])
        if trickbit_store[k] == 1:
            circuit = Circuit(2 * n + 3)
            X | circuit(trickbit)
            simulator.run(circuit, use_previous=True)

    # for idx in x_reg: Measure | circuit(idx)
    trickbit_store.reverse()
    logging.info(f'\tphi~ (approximately s/r) in binary form is {trickbit_store}')
    # continued fraction procedure
    phi_ = sum([(trickbit_store[i] * 1. / (1 << (i + 1))) for i in range(t)])
    logging.info(f'\tphi~ (approximately s/r) in decimal form is {phi_}')
    if phi_ == 0.0:
        logging.info('\torder_finding failed: phi~ = 0')
        return 0
    frac = Fraction(phi_).limit_denominator(N)
    logging.info(f'\tContinued fraction expansion of phi~ is {frac}')
    return frac.denominator
