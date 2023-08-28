import numpy as np
from sympy import symbols
from sympy.logic.boolalg import to_dnf

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.tools.exception.algorithm import *


class FRQI:
    """FRQI encoding for encoding classical image data into quantum circuits.

    For a 2^n x 2^n image, the number of qubits required for FRQI is 2n + 1 (2n position qubits + 1 color qubit).

    References:
        https://link.springer.com/article/10.1007/s11128-010-0177-y
    """

    def __init__(self, grayscale: int = 2):
        """Initialize a FRQI instance.

        Args:
            grayscale (int, optional): The grayscale of the input images. Defaults to 2.
        """

        self._grayscale = grayscale
        self._N = None
        self._n_qubits = None
        self._n_pos_qubits = None
        self._n_color_qubits = 1
        self._q_state = None

    def __str__(self):
        return "FRQI(n_qubits={}, grayscale={})".format(self._n_qubits, self._grayscale)

    def __call__(self, img, use_qic=False):
        """Call FRQI for a given image.

        Args:
            img (np.ndarray): The input image.
            use_qic (bool, optional): Whether to use Quantum Image Commpression. Defaults to False.

        Returns:
            Circuit: Quantum circuit form of the image.
        """

        img = self._img_preprocess(img, flatten=True)

        # step 1: |0> -> |H>
        circuit = Circuit(self._n_qubits)
        for qid in range(self._n_pos_qubits):
            H | circuit(qid)

        # step 2: |H> -> |I>
        if use_qic:
            frqi_circuit = self._construct_qic_circuit(img, rotate=True)
        else:
            frqi_circuit = self._construct_circuit(img, rotate=True)
        frqi_circuit | circuit

        return circuit

    def _img_preprocess(self, img, flatten=True):
        if ((img <= 1.0) & (img > 0.0)).any():
            img *= self._grayscale - 1
        img = img.astype(np.int64)
        assert (
            np.unique(img).shape[0] <= self._grayscale
            and np.max(img) <= self._grayscale
            and img.shape[0] == img.shape[1]
        ), EncodingError("Invalid image.")
        self._N = img.shape[0] * img.shape[1]
        self._n_pos_qubits = int(np.log2(self._N))
        assert 1 << self._n_pos_qubits == self._N, EncodingError(
            "The image size should be 2^n x 2^n."
        )
        self._q_state = [0] * self._n_pos_qubits
        self._n_qubits = self._n_pos_qubits + self._n_color_qubits
        if flatten:
            img = img.flatten()
        return img

    def _construct_circuit(self, img: np.ndarray, rotate: bool):
        circuit = CompositeGate(self._n_qubits)
        if not rotate:
            mc_gate = MultiControlGate(self._n_pos_qubits, GateType.x)
        for i in range(self._N):
            if img[i] < 1e-12:
                continue
            bin_pos = bin(i)[2:].zfill(self._n_pos_qubits)
            for qid in range(self._n_pos_qubits):
                if (bin_pos[qid] == "0" and self._q_state[qid] == 0) or (
                    bin_pos[qid] == "1" and self._q_state[qid] == 1
                ):
                    X | circuit(qid)
                    self._q_state[qid] = 1 - self._q_state[qid]
            if rotate:
                mc_gate = MultiControlGate(
                    self._n_pos_qubits,
                    GateType.ry,
                    params=[float(img[i] / (self._grayscale - 1) * np.pi)],
                )
                mc_gate | circuit(list(range(self._n_qubits)))
            else:
                bin_color = bin(img[i])[2:].zfill(self._n_color_qubits)
                for qid in range(self._n_color_qubits):
                    if bin_color[qid] == "1":
                        mct_qids = list(range(self._n_pos_qubits)) + [
                            self._n_pos_qubits + qid
                        ]
                        mc_gate | circuit(mct_qids)
        for qid in range(self._n_pos_qubits):
            if self._q_state[qid] == 1:
                X | circuit(qid)
                self._q_state[qid] = 1 - self._q_state[qid]
        return circuit

    def _construct_qic_circuit(self, img, rotate: bool, gid: int = 0):
        qic_circuit = CompositeGate(self._n_qubits)
        img_dict = self._get_img_dict(img, bin_val=True)
        for key in img_dict.keys():
            theta = float(key) / (self._grayscale - 1) * np.pi if rotate else None
            min_dnf = self._get_min_expression(img_dict[key])
            dnf_circuit = self._construct_dnf_circuit(min_dnf, gid, theta)
            dnf_circuit | qic_circuit
        for qid in range(self._n_pos_qubits):
            if self._q_state[qid] == 1:
                X | qic_circuit(qid)
                self._q_state[qid] = 1 - self._q_state[qid]
        return qic_circuit

    def _construct_dnf_circuit(self, min_dnf, gid: int = 0, theta: float = None):
        dnf_circuit = CompositeGate(self._n_qubits)
        cnf_list = self._split_dnf(min_dnf)
        if cnf_list == ["True"]:
            if theta is None:
                X | dnf_circuit(gid + self._n_pos_qubits)
            else:
                Ry(theta) | dnf_circuit(gid + self._n_pos_qubits)
            return dnf_circuit
        for i in range(len(cnf_list)):
            if i > 0:
                uniqueness_dnf = self._get_uniqueness_dnf(cnf_list[:i], cnf_list[i])
                uniqueness_dnf_circuit = self._construct_dnf_circuit(
                    uniqueness_dnf, gid, theta
                )
                uniqueness_dnf_circuit | dnf_circuit
            else:
                cnf_circuit = self._construct_cnf_circuit(
                    cnf_list[i], gid=gid, theta=theta,
                )
                cnf_circuit | dnf_circuit

        return dnf_circuit

    def _construct_cnf_circuit(self, cnf, gid=0, theta=None):
        cnf_circuit = CompositeGate(self._n_qubits)
        items = self._split_cnf(cnf)
        qids = self._get_cnf_qid(items)

        for item, qid in zip(items, qids):
            if (item[0] == "~" and self._q_state[qid] == 0) or (
                item[0] != "~" and self._q_state[qid] != 0
            ):
                X | cnf_circuit(qid)
                self._q_state[qid] = 1 - self._q_state[qid]

        mc_gate = (
            MultiControlGate(len(qids), GateType.x)
            if theta is None
            else MultiControlGate(len(qids), GateType.ry, params=[theta],)
        )
        mc_gate | cnf_circuit(qids + [gid + self._n_pos_qubits])
        return cnf_circuit

    def _get_uniqueness_dnf(self, pre_cnf_list, current_cnf):
        uniqueness_dnf = ""
        for cnf in pre_cnf_list:
            uniqueness_dnf += "~(" + cnf + ") & "
        uniqueness_dnf += "(" + current_cnf + ")"
        uniqueness_dnf = to_dnf(uniqueness_dnf, simplify=True, force=True)
        return uniqueness_dnf

    def _get_cnf_qid(self, cnf_items):
        idx_list = []
        for item in cnf_items:
            idx_list.append(int(item[item.index("_") + 1:]))
        return idx_list

    def _split_dnf(self, dnf):
        return str(dnf).replace("(", "").replace(")", "").split(" | ")

    def _split_cnf(self, cnf):
        return str(cnf).replace("(", "").replace(")", "").split(" & ")

    def _get_img_dict(self, img, bin_key=False, bin_val=False):
        img_dict = dict()
        for i in range(self._N):
            if img[i] < 1e-12:
                continue
            key = (
                bin(img[i])[2:].zfill(self._n_color_qubits) if bin_key else str(img[i])
            )
            val = bin(i)[2:].zfill(self._n_pos_qubits) if bin_val else i
            if key not in img_dict.keys():
                img_dict[key] = [val]
            else:
                img_dict[key].append(val)
        return img_dict

    def _get_boolen_expression(self, pixel):
        boolen_expression = ""
        x = symbols("x_0:" + str(self._N))
        for i in range(self._n_pos_qubits):
            boolen_expression += (
                "~" + str(x[i]) + " " if pixel[i] == "0" else str(x[i]) + " "
            )
            if i != self._n_pos_qubits - 1:
                boolen_expression += "& "
        return "( " + boolen_expression + ")"

    def _get_min_expression(self, pixels):
        boolen_expressions = ""
        for i in range(len(pixels)):
            boolen_expressions += self._get_boolen_expression(pixels[i])
            if i != len(pixels) - 1:
                boolen_expressions += " | "
        min_expression = to_dnf(boolen_expressions, simplify=True, force=True)
        return min_expression
