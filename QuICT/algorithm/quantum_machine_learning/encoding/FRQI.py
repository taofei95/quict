import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *

from QuICT.algorithm.quantum_machine_learning.utils.binary_reduction import (
    Binary_reduction,
)


class FRQI:
    def __init__(self, grayscale):
        self._grayscale = grayscale

    def multi_X_gate(self, act_qubit_list):
        composite_xgate = CompositeGate()
        for i in act_qubit_list:
            X & i | composite_xgate
        return composite_xgate

    def get_ctrl_list(self, bins, n_pos_qubits):
        """
        bins(int): reduced bins expression of 'position information'
        """
        bin_str_suffix = bin(bins)[-n_pos_qubits:]
        bin_str_prefixes = bin(bins).zfill(n_pos_qubits * 2 + 2)[2:-n_pos_qubits]
        zero_crtl_list = list()  
        # zero_crtl means flap the second qbit when first bit is 0 : 00-->01,01 -->00
        # crtl means flap the second qbit when first bit is 1 : 10-->11,11 -->10
        for j in range(len(bin_str_suffix)):
            if bin_str_suffix[j] == "0" and bin_str_prefixes[j] == "0":
                # supposed that position qubit before color qubit in the circuit
                zero_crtl_list.append(j)
        crtl_list = []
        for j in range(len(bin_str_suffix)):
            if bin_str_suffix[j] == "0":
                crtl_list.append(j)
        return zero_crtl_list, crtl_list

    def create_img_list(self, img):
        """
        this method used to group 'position information' by 'color information'
        """
        img_dic = dict()
        for i in range(len(img)):
            if img[i] not in img_dic:
                img_dic[str(img[i])] = [i]
            else:
                img_dic[str(img[i])].append(i)
        return img_dic

    def encoding(self, img):
        img = img.flatten()
        img_theta = img / (self._grayscale - 1) * np.pi
        N = img.shape[0]
        n_pos_qubits = int(np.log2(N))
        assert 1 << n_pos_qubits == N
        n_qubits = n_pos_qubits + 1

        img_dict = self.create_img_list(img_theta)

        circuit = Circuit(n_qubits)
        for qid in range(n_pos_qubits):
            H | circuit(qid)

        for item in img_dict:
            #  do binary reduce to 'position information'
            img_dict[item] = Binary_reduction(img_dict[item], n_pos_qubits)
            for jtem in img_dict[item]:
                zero_crtl_list, crtl_list = self.get_ctrl_list(jtem, n_pos_qubits)
                multi_x_gate = self.multi_X_gate(zero_crtl_list)
                multi_x_gate | circuit

                mcr = MultiControlRotation(GateType.ry, float(item))
                gates = mcr(control=crtl_list, target=n_pos_qubits)
                gates | circuit

                multi_x_gate | circuit

        return circuit

