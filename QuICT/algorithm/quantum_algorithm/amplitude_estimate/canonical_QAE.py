"""
canonical Quantum Amplitude Estimation \
in "Quantum Amplitude Amplification and Estimation"
see arXiv:quant-ph/0005055
"""

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate, Swap, H, Measure, IQFT

from QuICT.algorithm.quantum_algorithm.amplitude_estimate.utility import (
    OracleInfo,
    StatePreparationInfo,
)


def construct_circuit(
    eps=0.1,
    oracle: OracleInfo = None,
    state_preparation: StatePreparationInfo = None,
):
    if oracle is None:
        raise AssertionError("oracle info must be given")
    if state_preparation is None:
        state_preparation = StatePreparationInfo(n=oracle.n)
    assert state_preparation.n == oracle.n
    n = oracle.n
    # see Theorem 12, case k=1
    m = int(np.ceil(np.log2(2 * np.pi / (np.sqrt(1 + 4 * eps) - 1))))
    n_ancilla = max([1, oracle.n_ancilla, state_preparation.n_ancilla])
    trickbits = list(range(m))
    indices = list(range(m, m + n))
    ancilla = list(range(m + n, m + n + n_ancilla))

    cgate = CompositeGate()
    # state prepare
    state_preparation.A(n) | cgate(indices + ancilla)
    # Walsh-Hadamard transform
    for i in range(0, m):
        H | cgate(i)
    # controlled-Q^{2^j}
    Q_gate = CompositeGate()
    if oracle.custom_grover_operator is not None:
        Q_gate = oracle.custom_grover_operator
    else:
        oracle.S_chi(n, controlled=True) | Q_gate
        state_preparation.A_dagger(n, controlled=True) | Q_gate
        state_preparation.S_0(n, controlled=True) | Q_gate
        state_preparation.A(n, controlled=True) | Q_gate
    for j in range(m):
        for _ in range(1 << (m - 1 - j)):
            Q_gate | cgate([j] + list(range(m, m + n + n_ancilla)))
    # IQFT
    for k in range(len(trickbits) // 2):
        Swap | cgate([trickbits[k], trickbits[len(trickbits) - 1 - k]])
    IQFT.build_gate(m) | cgate(trickbits)
    for k in range(len(trickbits) // 2):
        Swap | cgate([trickbits[k], trickbits[len(trickbits) - 1 - k]])

    return cgate, {"trickbits": trickbits, "m": m, "n": n, "n_ancilla": n_ancilla}


def amplitude_estimate(
    eps=0.1,
    oracle: OracleInfo = None,
    state_preparation: StatePreparationInfo = None,
    simulator=None,
):
    cgate, info = construct_circuit(eps, oracle, state_preparation)
    from types import SimpleNamespace

    info = SimpleNamespace(**info)
    circ = Circuit(info.m + info.n + info.n_ancilla)
    cgate | circ
    for idx in info.trickbits:
        Measure | circ(idx)
    simulator.run(circ)
    y = int(circ[info.trickbits])
    return np.sin(np.pi * y / (1 << info.m)) ** 2
