from QuICT.core import Circuit
from QuICT.core.gate import *

from QuICT.algorithm.quantum_machine_learning.utils import Ansatz
from QuICT.algorithm.quantum_machine_learning.utils.gate_tensor import *


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

    def encoding(self, img, grayscale=2):
        img = img.flatten()
        img_theta = img / (grayscale - 1) * np.pi
        N = img.shape[0]
        n_pos_qubits = int(np.log2(N))
        assert 1 << n_pos_qubits == N
        n_qubits = n_pos_qubits + 1

        self._circuit = Circuit(n_qubits)
        for qid in range(n_pos_qubits):
            H | self._circuit(qid)

        for i in range(N):
            if i > 0:
                bin_str = bin((i - 1) ^ i)[2:].zfill(n_pos_qubits)
                for qid in range(n_pos_qubits):
                    if bin_str[qid] == "1":
                        X | self._circuit(qid)

            mcr = MultiControlRotation(GateType.ry, float(img_theta[i]))
            gates = mcr(control=list(range(n_pos_qubits)), target=n_pos_qubits)
            gates | self._circuit


class NEQR:
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

    def encoding(self, img):
        n_color_qubits = np.log(256)
        color_list = list()
        img = img.flatten()
        N = img.shape[0]
        n_pos_qubits = int(np.log2(N))
        assert 1 << n_pos_qubits == N
        n_qubits = n_pos_qubits + n_color_qubits + 1
        gate_ncnot = MultiControlToffoli()
        self._circuit = Circuit(n_qubits)

        for i in range(255):
            tmp = set()
            color_list.append(tmp)
        for item in range(N):
            bin_str = bin(item)[2:].zfill(n_pos_qubits)
            color_list[img[item]].add(bin_str)

        tp = list()
        tp.append(set())
        tp.append(set())
        old = 0
        new = 1
        for sets in color_list:
            tp[old] = sets.copy()
            while 1:
                for pos_bin in sets:
                    if pos_bin in tp[new]:
                        continue
                for pos_jin in sets:
                    if pos_bin == pos_jin:
                        continue
                    mod = int(pos_bin, 2) ^ int(pos_jin, 2)
                    if bin(mod).count("1") != 1:
                        break
                    tp[new].add(bin(int(pos_bin, 2) - mod))
                    tp[new].add(bin(int(pos_jin, 2) - mod))
                    tp[old].remove(pos_jin)
                    break
                temep = old
                old = new
                new = temep
                tp[new].clear()
                if not tp[old]:
                    sets.clear()
                    for item in tp[new]:
                        sets.add(item)
                    break
            for pos_bin in sets:
                img[int(pos_bin, 2)] = color_list.index(sets)

        for qid in range(n_pos_qubits):
            H | self._circuit(qid)

        for i in range(n_color_qubits):  # bin_express
            for qid in range(N):
                if bin(img[qid])[2:][i] == "0":
                    pass
                else:
                    gate_ncnot(control=n_pos_qubits) | self.circuit(i + n_pos_qubits)
