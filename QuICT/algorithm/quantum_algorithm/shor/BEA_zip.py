# Author  : Peng Sirui

"""
The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
St´ephane Beauregard in "Circuit for Shor’s algorithm using 2n+3 qubits"\
"""

import logging
from math import pi, gcd
import numpy as np
from fractions import Fraction

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.qcda.synthesis.arithmetic.bea import *
from QuICT.core.operator import Trigger, CheckPoint
from QuICT.simulation.state_vector import CircuitSimulator
from .utility import *


def construct_circuit(a: int, N: int, eps: float = 1 / 10):
    # phase estimation procedure
    n = int(np.ceil(np.log2(N + 1)))
    t = int(2 * n + 1 + np.ceil(np.log(2 + 1 / (2 * eps))))
    logging.info(f"\tcircuit construction begin: circuit: n = {n} t = {t}")

    circuit = Circuit(2 * n + 3)
    x_reg = [i for i in range(n + 1, 2 * n + 1)]
    trickbit = [2 * n + 1]

    # cgates[measured qubit index][added gate index][measured result]
    cgates = [
        [[CompositeGate() for measured in range(2)] for j in range(t + 1)]
        for i in range(t)
    ]
    # triggers[measured qubit index][added gate index]
    triggers = [
        [
            Trigger(1, [cgates[i][j][measured] for measured in range(2)])
            for j in range(t + 1)
        ]
        for i in range(t)
    ]
    triggers_reset = list(range(t))
    # checkpoints[measured qubit index]
    checkpoints = [CheckPoint() for i in range(t)]

    # triggers!!!
    for k in range(t):  # end at k-th checkpoints
        for i in range(k):  # start at i-th qubits (compressed)
            cpc = checkpoints[k].get_child()
            Rz(-np.pi / (1 << (k - i))) | cgates[i][k - 1 - i][1](trickbit)
            cpc | cgates[i][k - 1 - i][1]
    # triggers for reset!!!
    for i in range(t):
        cgate = CompositeGate()
        X | cgate(trickbit)
        triggers_reset[i] = Trigger(1, [CompositeGate(), cgate])

    # subcircuit: init \ket{1}\ket{0}
    X | circuit(x_reg[n - 1])

    history_indices = []
    ptr = 0
    for k in range(t):
        # subcircuit CUa
        H | circuit(trickbit)
        gate_pow = pow(a, 1 << (t - 1 - k), N)
        BEACUa.execute(n, gate_pow, N) | circuit
        # subcircuit: semi-classical QFT
        checkpoints[k] | circuit
        H | circuit(trickbit)
        # subcircuit: measure & reset trickbit
        for i in range(t - k):
            if i == 0:
                history_indices.append(ptr)
            triggers[k][i] | circuit(trickbit)
            ptr += 1
        triggers_reset[k] | circuit(trickbit)
        ptr += 1
    return circuit, trickbit + history_indices[::-1]


def order_finding(a: int, N: int, eps: float = 1 / 10, simulator=CircuitSimulator()):
    """
    The (2n+3)-qubit circuit used in the Shor algorithm is designed by \
    Stephane Beauregard in "Circuit for Shor's algorithm using 2n+3 qubits"

    Quantum algorithm to compute the order of a (mod N), when gcd(a,N)=1.
    """
    # phase estimation procedure
    n = int(np.ceil(np.log2(N + 1)))
    t = int(2 * n + 1 + np.ceil(np.log(2 + 1 / (2 * eps))))
    logging.info(f"\torder_finding begin: circuit: n = {n} t = {t}")
    trickbit_store = [0] * t

    b_reg = [i for i in range(n + 1)]
    x_reg = [i for i in range(n + 1, 2 * n + 1)]
    trickbit = [2 * n + 1]
    qreg_low = [2 * n + 2]
    # subcircuit: init \ket{1}\ket{0}
    circuit = Circuit(2 * n + 3)
    X | circuit(x_reg[n - 1])
    simulator.run(circuit)

    for k in range(t):
        # subcircuit CUa
        circuit = Circuit(2 * n + 3)
        H | circuit(trickbit)
        gate_pow = pow(a, 1 << (t - 1 - k), N)
        BEACUa.execute(n, gate_pow, N) | circuit
        simulator.run(circuit, use_previous=True)
        # subcircuit: semi-classical QFT
        circuit = Circuit(2 * n + 3)
        for i in range(k):
            if trickbit_store[i]:
                Rz(-pi / (1 << (k - i))) | circuit(trickbit)
        H | circuit(trickbit)
        simulator.run(circuit, use_previous=True)
        circuit = Circuit(2 * n + 3)
        for idx in b_reg + trickbit + qreg_low:
            Measure | circuit(idx)
        simulator.run(circuit, use_previous=True)
        # subcircuit: measure & reset trickbit
        assert int(circuit[qreg_low]) == 0 and int(circuit[b_reg]) == 0
        logging.info(f"\tthe {k}th trickbit measured to be {int(circuit[trickbit])}")
        trickbit_store[k] = int(circuit[trickbit])
        if trickbit_store[k] == 1:
            circuit = Circuit(2 * n + 3)
            X | circuit(trickbit)
            simulator.run(circuit, use_previous=True)

    # for idx in x_reg: Measure | circuit(idx)
    trickbit_store.reverse()
    logging.info(f"\tphi~ (approximately s/r) in binary form is {trickbit_store}")
    # continued fraction procedure
    phi_ = sum([(trickbit_store[i] * 1.0 / (1 << (i + 1))) for i in range(t)])
    logging.info(f"\tphi~ (approximately s/r) in decimal form is {phi_}")
    if phi_ == 0.0:
        logging.info("\torder_finding failed: phi~ = 0")
        return 0
    frac = Fraction(phi_).limit_denominator(N)
    logging.info(f"\tContinued fraction expansion of phi~ is {frac}")
    return frac.denominator
