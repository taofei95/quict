"""
Compute the disentangler of Pauli operators (O, O')
"""

class PauliOperator(object):
    """
    Pauli operator is a list of (I, X, Y, Z) with length n, which operates on n qubits.
    """


class PauliDisentangler(object):
    """
    For anti-commuting n-qubit Pauli operators O and O', there exists a Clifford circuit L∈C_n
    such that L^(-1) O L = X_1, L^(-1) O' L = Z_1.
    L is referred to as a disentangler for the pair (O, O').

    Reference:
        https://arxiv.org/abs/2105.02291
    """
