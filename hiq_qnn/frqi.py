import numpy as np
from mindquantum.core.circuit import Circuit, controlled
from mindquantum.core.gates import H, ZZ, RX, X, RY
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator


class HIQFRQI:
    def __init__(self, num_pixels):
        self._num_pixels = num_pixels

    def __call__(self):
        num_pos_qubit = int(np.log2(self._num_pixels))
        circuit = Circuit()
        circuit.un(H, list(range(num_pos_qubit)))
        for t, pos in enumerate(range(2 ** num_pos_qubit)):
            for k, c in enumerate("{0:0b}".format(pos).zfill(num_pos_qubit)):
                if c == "0":
                    circuit.x(k)
            circuit.ry(
                {f"ry{t}": np.pi}, num_pos_qubit, [i for i in range(num_pos_qubit)]
            )
            for k, c in enumerate("{0:0b}".format(pos).zfill(num_pos_qubit)):
                if c == "0":
                    circuit.x(k)

        return circuit

    def _img_preprocess(self, img, flatten=True):
        if ((img <= 1.0) & (img > 0.0)).any():
            img *= self._grayscale - 1
        img = img.astype(np.int64)
        assert (
            np.unique(img).shape[0] <= self._grayscale
            and np.max(img) <= self._grayscale
            and img.shape[0] == img.shape[1]
        )
        self._N = img.shape[0] * img.shape[1]
        self._n_pos_qubits = int(np.log2(self._N))
        assert 1 << self._n_pos_qubits == self._N
        self._q_state = [0] * self._n_pos_qubits
        self._n_qubits = self._n_pos_qubits + self._n_color_qubits
        if flatten:
            img = img.flatten()
        return img


if __name__ == "__main__":
    # frqi = HIQFRQI(2)
    # img = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])
    # circuit = frqi(img)
    # circuit.svg()
    circuit = Circuit()
    mc_gate = RY(np.pi).on(0, [1, 2, 3, 4])
    circuit += mc_gate
    np.set_printoptions(precision=2, threshold=np.inf, suppress=True)
    print(circuit.matrix().real)
