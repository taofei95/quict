import numpy as np

from .FRQI import FRQI
from QuICT.core import Circuit
from QuICT.core.gate import *


class NEQR(FRQI):
    """NEQR encoding for encoding classical image data into quantum circuits."""

    def __init__(self, grayscale: int = 2):
        super(NEQR, self).__init__(grayscale)
        self._n_color_qubits = int(np.log2(grayscale))
        assert 1 << self._n_color_qubits == grayscale

    def __call__(self, img, use_qic=True):
        img = self._img_preprocess(img, flatten=False)

        # step 1: |0> -> |H>
        circuit = Circuit(self._n_qubits)
        for qid in range(self._n_pos_qubits):
            H | circuit(qid)

        # step 2: |H> -> |I>
        if use_qic:
            neqr_circuit = Circuit(self._n_qubits)
            groups = self._get_groups(img)
            for i in range(self._n_color_qubits):
                sub_circuit = self._construct_qic_circuit(
                    groups[i], gid=i, drop_zerokey=True
                )
                sub_circuit | neqr_circuit(list(range(self._n_qubits)))
        else:
            neqr_circuit = self._construct_neqr_circuit(img)
        neqr_circuit | circuit(list(range(self._n_qubits)))
        return circuit

    def _construct_neqr_circuit(self, img):
        neqr_circuit = Circuit(self._n_qubits)
        mct = MultiControlToffoli()
        n = int(self._n_pos_qubits / 2)
        q_state = [1] * self._n_pos_qubits
        for y in range(img.shape[0]):
            bin_y = bin(y)[2:].zfill(n)
            # [pos-y] control qubits
            for qid in range(n):
                if (bin_y[qid] == "0" and q_state[qid + n] != 0) or (
                    bin_y[qid] == "1" and q_state[qid + n] != 1
                ):
                    X | neqr_circuit(qid + n)
                    q_state[qid + n] = 1 - q_state[qid + n]

            for x in range(img.shape[1]):
                if img[y, x] == 0:
                    continue
                bin_x = bin(x)[2:].zfill(n)
                bin_color = bin(img[y, x])[2:].zfill(self._n_color_qubits)
                # [pos-x] control qubits
                for qid in range(n):
                    if (bin_x[qid] == "0" and q_state[qid] != 0) or (
                        bin_x[qid] == "1" and q_state[qid] != 1
                    ):
                        X | neqr_circuit(qid)
                        q_state[qid] = 1 - q_state[qid]
                # [color] target qubits
                for qid in range(self._n_color_qubits):
                    if bin_color[qid] == "1":
                        mct_qids = list(range(self._n_pos_qubits)) + [
                            self._n_pos_qubits + qid
                        ]
                        mct(self._n_pos_qubits) | neqr_circuit(mct_qids)
        for qid in range(self._n_pos_qubits):
            if q_state[qid] == 0:
                X | neqr_circuit(qid)
        return neqr_circuit

    def _get_groups(self, img):
        img_dict = self._get_img_dict(img, bin_key=True)
        groups = np.zeros((self._n_color_qubits, self._N), dtype=np.bool_)
        for i in range(self._n_color_qubits):
            for bin_color in img_dict.keys():
                if bin_color[i] == "1":
                    for pixel in img_dict[bin_color]:
                        groups[i][pixel] = 1
        return groups


if __name__ == "__main__":
    from QuICT.simulation.state_vector import StateVectorSimulator
    import time

    np.set_printoptions(threshold=np.inf)

    neqr = NEQR(256)
    # img = np.array(
    #     [
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    #         [0.0, 0.0, 0.0, 0.12156863, 0.74509805, 0.74509805, 0.0, 0.0],
    #         [0.0, 0.0, 0.08235294, 0.7176471, 0.3882353, 0.98039216, 0.13333334, 0.0,],
    #         [0.0, 0.0, 0.0, 0.0, 0.67058825, 0.9882353, 0.0, 0.0,],
    #         [0.0, 0.0, 0.27058825, 0.7882353, 0.8117647, 0.46666667, 0.0, 0.0,],
    #         [0.0, 0.0, 0.01960784, 0.0, 0.41568628, 0.4627451, 0.0, 0.0,],
    #         [0.0, 0.49411765, 0.9882353, 0.9607843, 0.6509804, 0.0, 0.0, 0.0,],
    #         [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,],
    #     ]
    # )
    # neqr = NEQR(256)
    img = np.array([[0, 100], [200, 255]])
    circuit = neqr(img, use_qic=False)
    # circuit.gate_decomposition(decomposition=False)
    # circuit.draw(filename="neqr")
    simulator = StateVectorSimulator(device="GPU")
    start = time.time()
    sv = simulator.run(circuit)
    print(time.time() - start)
    circuit.gate_decomposition(decomposition=False)
    circuit.draw(filename="neqr")
    # mct = MultiControlToffoli()
    # circuit = Circuit(5)
    # mct(3) | circuit([0, 1, 2, 3])

    # neqr._img_preprocess(img)
    # neqr._construct_dnf_circuit(["01", "10", "11"], 1)
