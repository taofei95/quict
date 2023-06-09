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
        img = self._img_preprocess(img, flatten=True)

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
                    groups[i], rotate=False, gid=i
                )
                sub_circuit | neqr_circuit(list(range(self._n_qubits)))
        else:
            neqr_circuit = self._construct_circuit(img, rotate=False)
        neqr_circuit | circuit(list(range(self._n_qubits)))
        return circuit

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

    neqr = NEQR(2)
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
    # img = np.array([[0, 100], [200, 255]])
    img = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0,]])
    circuit = neqr(img, use_qic=False)
    # circuit.gate_decomposition(decomposition=False)
    # circuit.draw(filename="neqr")
    simulator = StateVectorSimulator(device="GPU")
    start = time.time()
    sv = simulator.run(circuit)
    print(sv)
    circuit.gate_decomposition(decomposition=False)
    circuit.draw(filename="neqr")
    # mct = MultiControlToffoli()
    # circuit = Circuit(5)
    # mct(3) | circuit([0, 1, 2, 3])

    # neqr._img_preprocess(img)
    # neqr._construct_dnf_circuit(["01", "10", "11"], 1)
