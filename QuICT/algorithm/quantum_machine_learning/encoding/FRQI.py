import numpy as np
from sympy import symbols
from sympy.logic.boolalg import to_dnf

from QuICT.core import Circuit
from QuICT.core.gate import *


class FRQI:
    """FRQI encoding for encoding classical image data into quantum circuits."""

    def __init__(self, grayscale: int = 2):
        self._grayscale = grayscale

    def __call__(self, img, use_qic=True):
        img = img.flatten()
        assert np.unique(img).shape[0] == self._grayscale
        N = img.shape[0]

        img_theta = img / (self._grayscale - 1) * np.pi
        N = img.shape[0]
        img_dict = self._get_img_dict(img_theta, N)

        n_pos_qubits = int(np.log2(N))
        assert 1 << n_pos_qubits == N
        n_pixel_qubits = 1
        n_qubits = n_pos_qubits + n_pixel_qubits

        # step 1: |0> -> |H>
        circuit = Circuit(n_qubits)
        for qid in range(n_pos_qubits):
            H | circuit(qid)

        # step 2: |H> -> |I>
        if use_qic:
            qic_circuit = self._construct_qic_circuit(
                img_dict, n_pos_qubits, n_pixel_qubits
            )
            qic_circuit | circuit(list(range(n_qubits)))
        else:
            for i in range(N):
                if i > 0:
                    bin_str = bin((i - 1) ^ i)[2:].zfill(n_pos_qubits)
                    for qid in range(n_pos_qubits):
                        if bin_str[qid] == "1":
                            X | circuit(qid)
                mcr = MultiControlRotation(GateType.ry, float(img_theta[i]))
                gates = mcr(control=list(range(n_pos_qubits)), target=n_pos_qubits)
                gates | circuit

        return circuit

    def _construct_qic_circuit(self, img_dict, n_pos_qubits, n_pixel_qubits):
        n_qubits = n_pos_qubits + n_pixel_qubits
        qic_circuit = Circuit(n_qubits)
        for key in img_dict.keys():
            dnf_circuit = self._construct_dnf_circuit(
                img_dict[key], key, n_pos_qubits, n_pixel_qubits
            )
            dnf_circuit | qic_circuit(list(range(n_qubits)))
        return qic_circuit

    def _construct_dnf_circuit(self, pixels, rotate, n_pos_qubits, n_pixel_qubits):
        n_qubits = n_pos_qubits + n_pixel_qubits
        dnf_circuit = Circuit(n_qubits)
        min_expression = self._get_min_expression(pixels, n_pos_qubits)
        cnf_list = self._split_dnf(min_expression)
        appeared_qids = {"+": set(), "-": set()}
        q_state = [0] * n_pos_qubits
        for i in range(len(cnf_list)):
            cnf_circuit, appeared_qids, q_state = self._construct_cnf_circuit(
                cnf_list[i], appeared_qids, q_state, n_qubits, rotate,
            )
            cnf_circuit | dnf_circuit(list(range(n_qubits)))
        return dnf_circuit

    def _construct_cnf_circuit(self, cnf, appeared_qids, q_state, n_qubits, rotate):
        cnf_circuit = Circuit(n_qubits)
        mcr = MultiControlRotation(GateType.ry, float(rotate))
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
        mcr(control=controls, target=n_qubits - 1) | cnf_circuit

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

    def _get_img_dict(self, img_theta, N):
        img_dict = dict()
        for i in range(N):
            if str(img_theta[i]) not in img_dict.keys():
                img_dict[str(img_theta[i])] = [i]
            else:
                img_dict[str(img_theta[i])].append(i)
        return img_dict

    def _get_boolen_expression(self, pixel, n_pos_qubits):
        boolen_expression = ""
        N = int(1 << n_pos_qubits)
        x = symbols("x_0:" + str(N))
        for i in range(n_pos_qubits):
            boolen_expression += (
                "~" + str(x[n_pos_qubits - 1 - i]) + " "
                if pixel[i] == "0"
                else str(x[n_pos_qubits - 1 - i]) + " "
            )
            if i != n_pos_qubits - 1:
                boolen_expression += "& "
        return "( " + boolen_expression + ")"

    def _get_min_expression(self, pixels, n_pos_qubits):
        boolen_expressions = ""
        pixels = [bin(pixel)[2:].zfill(n_pos_qubits) for pixel in pixels]
        for i in range(len(pixels)):
            boolen_expressions += self._get_boolen_expression(pixels[i], n_pos_qubits)
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
    # circuit.gate_decomposition(decomposition=False)
    # mct(3).draw(filename="mct")
