import sys
sys.path.append('/mnt/e/ICT/QuICT')

import numpy as np

from QuICT.qcda.synthesis.unitary_transform.two_qubit_transform import CartanKAKDecomposition

def generate_unitary():
    detM = 0
    while np.isclose(detM, 0, rtol=1.0e-13, atol=1.0e-13):
        A = np.random.randn(4, 4)
        B = np.random.randn(4, 4)
        M = A + 1j * B
        detM = np.linalg.det(M)
    U, _, _ = np.linalg.svd(M)
    return U

def test():
    for _ in range(20):
        U = generate_unitary()
        KAK = CartanKAKDecomposition(U)
        KAK.decompose()

if __name__ == '__main__':
    test()