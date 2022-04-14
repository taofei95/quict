import numpy as np


CGATE_LIST = []


def matrix_product_to_circuit(gate_matrix, gate_args, max_q: int, min_q: int = 0):
    """ Expand gate matrix with the number of qubits

    Args:
        gate (BasicGate): The quantum gate
        max_q (int): The qubits' number
        min_q (int, optional): The minimum qubit's number. Defaults to 0.

    Returns:
        np.array: the expanded gate's 2-D matrix
    """
    n = 1 << (max_q - min_q)
    xor = n - 1
    new_values = np.zeros((n, n), dtype=np.complex128)
    assert gate_matrix.shape == (1 << len(gate_args), 1 << len(gate_args))
    for arg in gate_args:
        assert arg >= 0 and arg < max_q and isinstance(arg, int)

    datas = np.zeros(n, dtype=int)
    for i in range(n):
        nowi = 0
        for t_idx, targ in enumerate(gate_args):
            assert targ >= min_q and targ < max_q
            k = (max_q - min_q) - 1 - (targ - min_q)
            if (1 << k) & i != 0:
                nowi += (1 << (len(gate_args) - 1 - t_idx))

        datas[i] = nowi

    for i in gate_args:
        xor = xor ^ (1 << (max_q - 1 - i))

    for i in range(n):
        nowi = datas[i]
        for j in range(n):
            if (i & xor) == (j & xor):
                nowj = datas[j]
                new_values[i][j] = gate_matrix[nowi][nowj]

    return new_values
