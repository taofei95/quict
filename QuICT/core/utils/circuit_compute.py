import numpy as np


def matrix_product_to_circuit(qubits, gate) -> np.ndarray:
    n = 1 << qubits
    xor = n - 1
    new_values = np.zeros((n, n), dtype=np.complex128)

    targs = np.array(gate.cargs + gate.targs, dtype=int)
    matrix = gate.matrix.reshape(1 << len(targs), 1 << len(targs))
    datas = np.zeros(n, dtype=int)
    for i in range(n):
        nowi = 0
        for kk in range(len(targs)):
            k = qubits - 1 - targs[kk]
            if (1 << k) & i != 0:
                nowi += (1 << (len(targs) - 1 - kk))

        datas[i] = nowi

    for i in targs:
        xor = xor ^ (1 << (qubits - 1 - i))

    for i in range(n):
        nowi = datas[i]
        for j in range(n):
            if (i & xor) == (j & xor):
                nowj = datas[j]
                new_values[i][j] = matrix[nowi][nowj]

    return new_values
