import numpy as np
from sympy import symbols
from sympy.logic.boolalg import to_dnf

from QuICT.core import Circuit
from QuICT.core.gate import *


class FRQI:
    """FRQI encoding for encoding classical image data into quantum circuits."""

    def __init__(self, grayscale: int = 2):
        self._grayscale = grayscale
        self._N = None
        self._n_qubits = None
        self._n_pos_qubits = None
        self._n_color_qubits = 1

    def __call__(self, img, use_qic=True):
        img = self._img_preprocess(img)
        img = img / (self._grayscale - 1) * np.pi

        # step 1: |0> -> |H>
        circuit = Circuit(self._n_qubits)
        for qid in range(self._n_pos_qubits):
            H | circuit(qid)

        # step 2: |H> -> |I>
        if use_qic:
            frqi_circuit = self._construct_qic_circuit(img)
        else:
            frqi_circuit = self._construct_frqi_circuit(img)
        frqi_circuit | circuit(list(range(self._n_qubits)))

        return circuit

    def _img_preprocess(self, img):
        img = img.flatten()
        if ((img < 1.0) & (img > 0.0)).any():
            img *= self._grayscale - 1
        img = img.astype(np.int64)
        assert (
            np.unique(img).shape[0] <= self._grayscale
            and np.max(img) <= self._grayscale
        )
        self._N = img.shape[0]
        self._n_pos_qubits = int(np.log2(self._N))
        assert 1 << self._n_pos_qubits == self._N
        self._n_qubits = self._n_pos_qubits + self._n_color_qubits
        return img

    def _construct_frqi_circuit(self, img):
        frqi_circuit = Circuit(self._n_qubits)
        for i in range(self._N):
            if i > 0:
                bin_str = bin((i - 1) ^ i)[2:].zfill(self._n_pos_qubits)
                for qid in range(self._n_pos_qubits):
                    if bin_str[qid] == "1":
                        X | frqi_circuit(qid)
            mcr = MultiControlRotation(GateType.ry, float(img[i]))
            gates = mcr(
                control=list(range(self._n_pos_qubits)), target=self._n_pos_qubits
            )
            gates | frqi_circuit
        return frqi_circuit

    def _construct_qic_circuit(self, img, gid=0, drop_zerokey=False):
        qic_circuit = Circuit(self._n_qubits)
        img_dict = self._get_img_dict(img, bin_val=True)
        for key in img_dict.keys():
            if drop_zerokey and key == "False":
                continue
            rotate = None if drop_zerokey else float(key)
            dnf_circuit = self._construct_dnf_circuit(img_dict[key], gid, rotate)
            dnf_circuit | qic_circuit(list(range(self._n_qubits)))
        return qic_circuit

    def _construct_dnf_circuit(self, pixels, gid=0, rotate=None):
        dnf_circuit = Circuit(self._n_qubits)
        min_expression = self._get_min_expression(pixels)
        cnf_list = self._split_dnf(min_expression)
        if cnf_list == ["True"]:
            if rotate is None:
                X | dnf_circuit(gid + self._n_pos_qubits)
            else:
                Ry(rotate) | dnf_circuit(gid + self._n_pos_qubits)
            return dnf_circuit
        appeared_qids = {"+": set(), "-": set()}
        q_state = [0] * self._n_pos_qubits
        for i in range(len(cnf_list)):
            cnf_circuit, appeared_qids, q_state = self._construct_cnf_circuit(
                cnf_list[i], appeared_qids, q_state, gid=gid, rotate=rotate,
            )
            cnf_circuit | dnf_circuit(list(range(self._n_qubits)))
        for qid in range(self._n_pos_qubits):
            if q_state[qid] == -1:
                X | dnf_circuit(qid)

        return dnf_circuit

    def _construct_cnf_circuit(self, cnf, appeared_qids, q_state, gid=0, rotate=None):
        cnf_circuit = Circuit(self._n_qubits)
        mc_gate = (
            MultiControlToffoli()
            if rotate is None
            else MultiControlRotation(GateType.ry, rotate)
        )
        items = self._split_cnf(cnf)
        qids = self._get_cnf_qid(items)
        controls = []

        not_appeared_qids = (
            (appeared_qids["+"] | appeared_qids["-"])
            - set(qids)
            - (appeared_qids["+"] & appeared_qids["-"])
        )

        for qid in not_appeared_qids:
            assert abs(q_state[qid]) > 0
            X | cnf_circuit(qid)
            q_state[qid] = -q_state[qid]
            controls.append(qid)

        for item, qid in zip(items, qids):
            if item[0] == "~":
                appeared_qids["-"].add(qid)
                if q_state[qid] != -1:
                    X | cnf_circuit(qid)
                q_state[qid] = -1
            else:
                appeared_qids["+"].add(qid)
                if q_state[qid] == -1:
                    X | cnf_circuit(qid)
                q_state[qid] = 1
            controls.append(qid)

        c_gate = (
            mc_gate(len(controls))
            if rotate is None
            else mc_gate(controls, gid + self._n_pos_qubits)
        )
        c_gate | cnf_circuit(controls + [gid + self._n_pos_qubits])

        return cnf_circuit, appeared_qids, q_state

    def _get_cnf_qid(self, cnf_items):
        idx_list = []
        for item in cnf_items:
            idx_list.append(int(item[item.index("_") + 1 :]))
        return idx_list

    def _split_dnf(self, dnf):
        return str(dnf).replace("(", "").replace(")", "").split(" | ")

    def _split_cnf(self, cnf):
        return str(cnf).replace("(", "").replace(")", "").split(" & ")

    def _get_img_dict(self, img, bin_key=False, bin_val=False):
        img_dict = dict()
        for i in range(self._N):
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
                "~" + str(x[self._n_pos_qubits - 1 - i]) + " "
                if pixel[i] == "0"
                else str(x[self._n_pos_qubits - 1 - i]) + " "
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
        min_expression = to_dnf(boolen_expressions, simplify=True)
        return min_expression


if __name__ == "__main__":
    from QuICT.simulation.state_vector import StateVectorSimulator
    import time

    frqi = FRQI(2)
    img = np.array([[1, 0, 1, 0], [1, 0, 0, 1], [0, 0, 1, 0], [1, 1, 0, 0,]])
    circuit = frqi(img)

    simulator = StateVectorSimulator(device="GPU")
    start = time.time()
    sv = simulator.run(circuit)
    # print(sv)
    print(time.time() - start)
    # circuit.gate_decomposition(decomposition=False)
    # mct = MultiControlToffoli()
    # circuit = Circuit(5)
    # mct(3) | circuit([0, 1, 2, 3])
    circuit.gate_decomposition(decomposition=False)
    circuit.draw(filename="frqi")
