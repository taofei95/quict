from typing import Union
import numpy as np


CGATE_LIST = []


def matrix_product_to_circuit(
    gate_matrix: np.ndarray,
    gate_args: Union[int, list],
    max_q: int
):
    """ Expand gate matrix with the number of qubits

    Args:
        gate_matrix (np.ndarray): The gate's matrix.
        gate_args Union[int, list]: The gate's qubit indexes.
        max_q (int): The qubits' number
        min_q (int, optional): The minimum qubit's number. Defaults to 0.
        gpu_output(bool, optional): Generate matrix in GPU or not. Default to False.

    Returns:
        np.array: the expanded gate's 2-D matrix
    """
    if isinstance(gate_args, int):
        gate_args = [gate_args]

    n = 1 << max_q
    xor = n - 1
    new_values = np.zeros((n, n), dtype=gate_matrix.dtype)
    assert gate_matrix.shape == (1 << len(gate_args), 1 << len(gate_args))
    for arg in gate_args:
        assert arg >= 0 and arg < max_q and isinstance(arg, int)

    datas = np.zeros(n, dtype=int)
    for i in range(n):
        nowi = 0
        for t_idx, targ in enumerate(gate_args):
            assert targ >= 0 and targ < max_q
            k = max_q - 1 - targ
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
