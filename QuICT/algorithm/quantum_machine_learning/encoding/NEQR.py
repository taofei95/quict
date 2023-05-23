import numpy as np
from sympy import symbols
from sympy.logic.boolalg import to_dnf

from FRQI import FRQI
from QuICT.core import Circuit
from QuICT.core.gate import *


class NEQR(FRQI):
    """NEQR encoding for encoding classical image data into quantum circuits."""

    def __init__(self, grayscale: int = 2):
        super(NEQR, self).__init__(grayscale)
        self._n_color_qubits = int(np.log2(grayscale))
        assert 1 << self._n_color_qubits == grayscale

    def __call__(self, img, use_qic=True):
        img = self._img_preprocess(img)

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
        for i in range(1, self._N):
            control_not = []
            bin_pos = bin(i)[2:].zfill(self._n_pos_qubits)
            for qid in range(self._n_pos_qubits):
                if bin_pos[qid] == "0":
                    X | neqr_circuit(qid)
                    control_not.append(qid)
            bin_color = bin(img[i])[2:].zfill(self._n_color_qubits)
            for qid in range(self._n_color_qubits):
                if bin_color[qid] == "1":
                    mct_qids = list(range(self._n_pos_qubits)) + [
                        self._n_pos_qubits + qid
                    ]
                    mct(self._n_pos_qubits) & mct_qids | neqr_circuit
            for qid in control_not:
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

    neqr = NEQR(2)
    img = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0,]])
    # neqr = NEQR(256)
    # img = np.array([[0, 100], [200, 255]])
    circuit = neqr(img)
    circuit.gate_decomposition(decomposition=False)
    circuit.draw(filename="neqr")
    # simulator = StateVectorSimulator(device="GPU")
    # start = time.time()
    # sv = simulator.run(circuit)
    # # print(sv)
    # print(time.time() - start)
    # circuit.gate_decomposition(decomposition=False)
    # mct = MultiControlToffoli()
    # circuit = Circuit(5)
    # mct(3) | circuit([0, 1, 2, 3])

    # neqr._img_preprocess(img)
    # neqr._construct_dnf_circuit(["01", "10", "11"], 1)
