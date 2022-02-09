import numpy as np


def matrix_product_to_circuit(gate, max_q, min_q: int = 0):
    n = 1 << (max_q - min_q)
    xor = n - 1
    new_values = np.zeros((n, n), dtype=np.complex128)

    targs = np.array(gate.cargs + gate.targs, dtype=int)
    matrix = gate.matrix.reshape(1 << len(targs), 1 << len(targs))
    datas = np.zeros(n, dtype=int)
    for i in range(n):
        nowi = 0
        for t_idx, targ in enumerate(targs):
            assert targ >= min_q and targ < max_q
            k = (max_q - min_q) - 1 - (targ - min_q)
            if (1 << k) & i != 0:
                nowi += (1 << (len(targs) - 1 - t_idx))

        datas[i] = nowi

    for i in targs:
        xor = xor ^ (1 << (max_q - 1 - i))

    for i in range(n):
        nowi = datas[i]
        for j in range(n):
            if (i & xor) == (j & xor):
                nowj = datas[j]
                new_values[i][j] = matrix[nowi][nowj]

    return new_values
