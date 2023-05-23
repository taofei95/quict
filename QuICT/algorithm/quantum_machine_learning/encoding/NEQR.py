import numpy as np

from .FRQI import FRQI
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.algorithm.quantum_machine_learning.utils.binary_reduction import (
    Binary_reduction,
)


class NEQR(FRQI):
    def create_img_list(self, img, n_color_qubits, N):
        img_list = []
        for i in range(n_color_qubits):
            img_list.append(list())
        for i in range(N):
            color_bit = bin(img[i])[2:]
            for j in range(len(color_bit)):
                if color_bit[j] == "1":
                    img_list[n_color_qubits - 1 - j].append(i)
        return img_list

    def encoding(self, img):
        img = img.flatten()
        N = img.shape[0]
        n_color_qubits = int(np.log2(self._grayscale))
        n_pos_qubits = int(np.log2(N))
        assert 1 << n_pos_qubits == N
        n_qubits = n_pos_qubits + n_color_qubits + 1

        MCT = MultiControlToffoli()
        circuit = Circuit(n_qubits)
        img_list = self.create_img_list(img, n_color_qubits, N)

        for qid in range(n_pos_qubits):
            H | circuit(qid)

        for i in range(n_color_qubits):
            img_list[i] = Binary_reduction(img_list[i], n_pos_qubits)
            for item in img_list[i]:
                zero_crtl_list, crtl_list = self.get_ctrl_list(item, n_pos_qubits)
                multi_x_gate = self.multi_X_gate(zero_crtl_list)
                multi_x_gate | circuit

                act_qubit_list = crtl_list.copy()
                act_qubit_list.append(i + n_pos_qubits)
                MCT(control=len(crtl_list)) | circuit(act_qubit_list)

                multi_x_gate | circuit

        return circuit
