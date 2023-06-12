
import numpy as np
from mindquantum.core.circuit import Circuit, controlled
from mindquantum.core.gates import H, ZZ, RX, X, RY
from mindquantum.core.operators import Hamiltonian, QubitOperator
from mindquantum.framework import MQAnsatzOnlyLayer
from mindquantum.simulator import Simulator


class FRQI:
    def __init__(self, grayscale: int = 2):
        self._grayscale = grayscale
        self._N = None
        self._n_qubits = None
        self._n_pos_qubits = None
        self._n_color_qubits = 1
        self._q_state = None

    def __call__(self, img):
        img = self._img_preprocess(img, flatten=True)

        # step 1: |0> -> |H>
        circuit = Circuit()
        for qid in range(self._n_pos_qubits):
            circuit += H.on(qid)

        # step 2: |H> -> |I>
        frqi_circuit = self._construct_circuit(img, rotate=True)
        circuit += frqi_circuit

        return circuit

    def _img_preprocess(self, img, flatten=True):
        if ((img < 1.0) & (img > 0.0)).any():
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

    def _construct_circuit(self, img: np.ndarray, rotate: bool):
        circuit = Circuit()
        for i in range(self._N):
            if img[i] == 0:
                continue
            bin_pos = bin(i)[2:].zfill(self._n_pos_qubits)
            for qid in range(self._n_pos_qubits):
                if (bin_pos[qid] == "0" and self._q_state[qid] == 0) or (
                    bin_pos[qid] == "1" and self._q_state[qid] == 1
                ):
                    circuit += X.on(qid)
                    self._q_state[qid] = 1 - self._q_state[qid]
            theta = float(img[i] / (self._grayscale - 1) * np.pi)
            mc_gate = RY(theta).on(self._n_pos_qubits, list(range(self._n_pos_qubits)))
            circuit += mc_gate

        for qid in range(self._n_pos_qubits):
            if self._q_state[qid] == 1:
                circuit += X.on(qid)
        return circuit


if __name__ == "__main__":
    frqi = FRQI(2)
    img = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0,]])
    circuit = frqi(img)
    circuit.svg()