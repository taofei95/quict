"""
Quantum Amplitude Estimation in "Amplitude estimation without phase estimation"
see arXiv:1904.10246[quant-ph]
"""

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate, Measure
from .utility import OracleInfo, StatePreparationInfo


def amplitude_estimate(
    eps=0.1,
    oracle: OracleInfo = None,
    state_preparation: StatePreparationInfo = None,
    simulator=None,
):
    """maximum likelyhood estimation. The list of (N,m) is \
    constructed from eps in EIS style where N_shot=100

    Args:
        eps (float, optional): allowed error. Defaults to 0.1.
        oracle (OracleInfo, optional): Defaults to None.
        state_preparation (StatePreparationInfo, optional): Defaults to None.
        simulator (Simulator, optional): Defaults to None.
    """
    if oracle is None:
        raise AssertionError("oracle info must be given")
    if state_preparation is None:
        state_preparation = StatePreparationInfo(n=oracle.n)
    assert state_preparation.n == oracle.n
    n = oracle.n
    n_ancilla = max([1, oracle.n_ancilla, state_preparation.n_ancilla])

    # list (N,m): N=100, m=[2**0...2**(M-1)]
    N_shot = 100
    M = max(3, np.log2((1 / eps) / N_shot))
    list_m = np.array([2 ** i for i in range(int(M))])

    # Q^{m[i]}
    Q_gate = CompositeGate()
    if oracle.custom_grover_operator is not None:
        Q_gate = oracle.custom_grover_operator
    else:
        oracle.S_chi(n) | Q_gate
        state_preparation.A_dagger(n) | Q_gate
        state_preparation.S_0(n) | Q_gate
        state_preparation.A(n) | Q_gate

    good_counts = np.zeros(shape=(len(list_m),))
    all_counts = np.ones(shape=(len(list_m),)) * N_shot
    for i, m in enumerate(list_m):
        circ = Circuit(n + n_ancilla)
        state_preparation.A(n) | circ
        for _ in range(m):
            Q_gate | circ
        for j in range(n):
            Measure | circ(j)
        h = 0
        for _ in range(N_shot):  # optimization: using `shot` in simulator
            simulator.run(circ)
            res = bin(int(circ[:n]))[2:].rjust(n, "0")
            if oracle.is_good_state(res):
                h += 1
        good_counts[i] = h

    # MLE
    from scipy.optimize import brute

    search_eps = 1e-10
    nevals = max(10000, int(np.pi / 2 * 1000 * 2 * (2 ** M)))

    def loglikelyhood(theta):
        fst_half = good_counts * np.log(np.sin((list_m * 2 + 1) * theta) ** 2)
        sec_half = (all_counts - good_counts) * np.log(
            np.cos((list_m * 2 + 1) * theta) ** 2
        )
        return -(np.sum(fst_half + sec_half))

    est_theta = brute(
        loglikelyhood,
        ranges=[[0 + search_eps, np.pi / 2 - search_eps]],
        Ns=nevals)[0]
    return np.sin(est_theta) ** 2
