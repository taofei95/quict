import numpy as np
import torch
from QuICT.algorithm.quantum_machine_learning.utils_v1 import Ansatz
from QuICT.algorithm.quantum_machine_learning.utils_v1.gate_tensor import *
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.algorithm.quantum_machine_learning.utils.binary_reduction import Binary_reduction


class Qubit:
    """Qubit encoding for encoding classical image data into quantum circuits."""

    @property
    def circuit(self):
        return self._circuit

    @property
    def ansatz(self):
        return self._ansatz

    def __init__(self, data_qubits, device=torch.device("cuda:0")):
        """The qubit encoding constructor.

        Args:
            data_qubits (int): The number of the data qubits.
            device (str, optional): The device to which the model is assigned.
                Defaults to "cuda:0".
        """
        self._data_qubits = data_qubits
        self._device = device
        self._circuit = None
        self._ansatz = None

    def encoding(self, img, circuit=False):
        """Encode the image as quantum ansatz using qubit encoding.
        One pixel corresponds to one qubit.

        Args:
            img (torch.Tensor): The classical image data.
            circuit (bool): If True, build data circuits at the same time.
                Defaults to False.
        """
        img = img.flatten()
        self._ansatz = Ansatz(self._data_qubits, device=self._device)
        for i in range(img.shape[0]):
            if img[i] == 1:
                self._ansatz.add_gate(X_tensor, i)
        if circuit:
            self._circuit = Circuit(self._data_qubits)
            for i in range(img.shape[0]):
                if img[i] == 1:
                    X | self._circuit(i)


class Amplitude:
    """Amplitude encoding for encoding classical image data into quantum circuits."""

    @property
    def circuit(self):
        return self._circuit

    @property
    def ansatz(self):
        return self._ansatz

    def __init__(self, data_qubits, device=torch.device("cuda:0")):
        """The amplitude encoding constructor.

        Args:
            data_qubits (int): The number of the data qubits.
            device (str, optional): The device to which the model is assigned.
                Defaults to "cuda:0".
        """
        self._data_qubits = data_qubits
        self._device = device
        self._circuit = None
        self._ansatz = None

    def encoding(self, img, circuit=False):
        """Encode the image as quantum ansatz using amplitude encoding."""
        raise NotImplementedError


class FRQI:
    @property
    def circuit(self):
        return self._circuit

    @property
    def ansatz(self):
        return self._ansatz

    def __init__(self, device=torch.device("cuda:0")):
        self._device = device
        self._circuit = None
        self._ansatz = None

    def multi_X_gate(self,act_qubit_list):
        composite_xgate = CompositeGate()
        for i in act_qubit_list:
            X & i |composite_xgate
        return composite_xgate

    def get_ctrl_list(self,bins,n_pos_qubits):
        bin_str_suffix = bin(bins)[-n_pos_qubits:]
        bin_str_prefixes = bin(bins).zfill(n_pos_qubits*2+2)[2: -n_pos_qubits]
        zero_crtl_list = list()
        for j in range(len(bin_str_suffix)):
            if bin_str_suffix[j] == '0' and bin_str_prefixes[j] == '0':
                zero_crtl_list.append(j)  # position qubit before color qubit in the circuit
        crtl_list = []
        for j in range(len(bin_str_suffix)):
            if bin_str_suffix[j] == '0':
                crtl_list.append(j)
        return zero_crtl_list,crtl_list
 
    def create_img_list(self,img):
        img_dic = dict()
        for i in range(len(img)):
            if img[i] not in img_dic:
                img_dic[str(img[i])] = [i]
            else:
                img_dic[str(img[i])].append(i) 
        return img_dic
    def encoding(self, img, grayscale=2,):
        img = img.flatten()
        img_theta = img / (grayscale - 1) * np.pi
        N = img.shape[0]
        n_pos_qubits = int(np.log2(N))
        assert 1 << n_pos_qubits == N
        n_qubits = n_pos_qubits + 1

        img_dict = self.create_img_list(img_theta)

        self._circuit = Circuit(n_qubits)
        for qid in range(n_pos_qubits):
            H | self._circuit(qid)

        for item in img_dict:
            img_dict[item] = Binary_reduction(img_dict[item],n_pos_qubits)
            for jtem in img_dict[item]:
                zero_crtl_list,crtl_list = self.get_ctrl_list(jtem,n_pos_qubits)
                multi_x_gate =self.multi_X_gate(zero_crtl_list)
                multi_x_gate |self._circuit

                mcr = MultiControlRotation(GateType.ry, float(item))
                gates = mcr(control=crtl_list, target=n_pos_qubits)
                gates | self._circuit

                multi_x_gate |self._circuit

class NEQR(FRQI):
    
    def create_img_list(self,img,n_color_qubits,N):
        img_list = []
        for i in range(n_color_qubits):
            img_list.append(list())
        for i in range(N):
            color_bit = bin(img[i])[2:]
            for j in range(len(color_bit)):
                if color_bit[j] == '1':
                    img_list[n_color_qubits-1-j].append(i)
        return img_list
    
   
    def encoding(self, img):
        n_color_qubits = int(np.log2(256))
        img = img.flatten()
        N = img.shape[0]
        n_pos_qubits = int(np.log2(N))
        assert 1 << n_pos_qubits >= N
        n_qubits = n_pos_qubits + n_color_qubits + 1
        gate_ncnot = MultiControlToffoli()
        self._circuit = Circuit(n_qubits)
        img_list = self.create_img_list(img,n_color_qubits,N)
        
        for qid in range(n_pos_qubits):
            H | self._circuit(qid)

        for i in range(n_color_qubits):
            img_list[i] = Binary_reduction(img_list[i],n_pos_qubits)
            for item in img_list[i]:
                zero_crtl_list,crtl_list = self.get_ctrl_list(item,n_pos_qubits)
                multi_x_gate =self.multi_X_gate(zero_crtl_list)
                multi_x_gate |self._circuit

                act_qubit_list = crtl_list.copy()
                act_qubit_list.append(i+n_pos_qubits)
                gate_ncnot = MultiControlToffoli()
                gate_ncnot(control=len(crtl_list))|self._circuit(act_qubit_list)

                multi_x_gate |self._circuit
