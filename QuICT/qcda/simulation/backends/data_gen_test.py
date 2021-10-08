#!/usr/bin/env python3

from random import choices, randint, sample, shuffle, uniform
from random import choice
from scipy.stats import unitary_group

from QuICT.core import *
from QuICT.algorithm import Amplitude
from math import sqrt, pi


def format_cpx(n):
    return '{0}{1}{2}j'.format(n.real, '+-'[int(n.imag < 0)], abs(n.imag))


diag_1_gates = {GATE_ID["Rz"]}
diag_2_gates = {}
ctrl_diag_gates = {GATE_ID["Crz"], GATE_ID["CU1"]}

ctrl_unitary_gate = {GATE_ID["CU3"]}


def out_circuit_to_file(qubit_num: int, f_name: str, circuit: Circuit):
    with open(f_name, 'w') as f:
        print(qubit_num, file=f)

        for gate in circuit.gates:
            gate: BasicGate
            if gate.type() == GATE_ID["X"]:
                print(f"special_x {gate.targ}", file=f)
            elif gate.type() == GATE_ID["H"]:
                print(f"special_h {gate.targ}", file=f)
            elif gate.type() in diag_1_gates:  # all 1-bit diagonal gates
                print(f"diag_1 {gate.targ} "
                      f"{format_cpx(gate.matrix[0, 0])} {format_cpx(gate.matrix[1, 1])}", file=f)
            elif gate.type() in diag_2_gates:  # all 2-bit diagonal gates (not supported now)
                NotImplementedError("No 2-bit diagonal gates for now")
            elif gate.type() in ctrl_diag_gates:
                print(f"ctrl_diag {gate.carg} {gate.targ} "
                      f"{format_cpx(gate.compute_matrix[2, 2])} {format_cpx(gate.compute_matrix[3, 3])}", file=f)
            else:
                if len(gate.affectArgs) == 1:  # all 1-bit unitary gates
                    print(f"unitary_1 {gate.targ} "
                          f"{format_cpx(gate.compute_matrix[0, 0])} {format_cpx(gate.compute_matrix[0, 1])} "
                          f"{format_cpx(gate.compute_matrix[1, 0])} {format_cpx(gate.compute_matrix[1, 1])}", file=f)
                elif len(gate.affectArgs) == 2:  # all 2-bit unitary gates
                    res_str = f"unitary_2 {gate.affectArgs[0]} {gate.affectArgs[1]} "
                    for i in range(4):
                        for j in range(4):
                            res_str += f"{format_cpx(gate.compute_matrix[i, j])} "
                    print(res_str, file=f)
                else:
                    NotImplementedError("No support for gate >= 3 bit for now")

        print("__TERM__", file=f)

        res = Amplitude.run(circuit)
        for val in res:
            if abs(val - 0) <= 1e-8:
                print("0+0j", file=f)
            else:
                print(format_cpx(val), file=f)


def rand_unitary_gate(qubit_num):
    mat = unitary_group.rvs(dim=1 << qubit_num)
    return Unitary(mat)


def manual_rand_unitary_gate(qubit_num):
    if qubit_num == 1:
        theta = random.random() * 2 * pi
        return random.choice([Rx, Ry])(theta)
    elif qubit_num == 2:
        return random.choice([
            Rxx(uniform(0, 2 * pi)),
            Ryy(uniform(0, 2 * pi)),
            Rzz(uniform(0, 2 * pi)),
            FSim([uniform(0, 2 * pi), uniform(0, 2 * pi)])
        ])


def main():
    qubit_num = 18
    circuit = Circuit(qubit_num)

    for i in range(qubit_num):
        H | circuit(i)

    for _ in range(qubit_num * 10):
        n = randint(1, 2)
        gate = manual_rand_unitary_gate(n)
        if n == 1:
            gate | circuit(randint(0, qubit_num - 1))
        else:
            gate | circuit(sample(range(0, qubit_num), 2))

    out_circuit_to_file(qubit_num, "u.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)

    for _ in range(30):
        Rz(uniform(0, 3.14)) | circuit(randint(0, qubit_num - 1))
    out_circuit_to_file(qubit_num, "diag.txt", circuit)
    circuit.clear()

    # for i in range(qubit_num):
    #     H | circuit(i)
    # # for _ in range(100):
    # #     lst = sample(range(0, qubit_num), 2)
    # #     CRz(uniform(0, 3.14)) | circuit([lst[0], lst[1]])
    # # X | circuit(qubit_num-1)
    # # X | circuit(qubit_num-3)
    # # CRz(pi) | circuit([qubit_num-3, qubit_num-1])
    # for _ in range(100):
    #     lst = sample(range(0, qubit_num), 2)
    #     rand_unitary_gate(2) | circuit([lst[0], lst[1]])
    #
    # out_circuit_to_file(qubit_num, "u2.txt", circuit)
    # circuit.clear()

    QFT(qubit_num).build_gate() | circuit
    out_circuit_to_file(qubit_num, "qft.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)

    out_circuit_to_file(qubit_num, "h.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)
    for _ in range(qubit_num * 40):
        lst = sample(range(0, qubit_num), 2)
        shuffle(lst)
        i = lst[0]
        j = lst[1]
        CRz(uniform(0, 3.14)) | circuit([i, j])

    out_circuit_to_file(qubit_num, "crz.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)
    for _ in range(qubit_num * 40):
        lst = sample(range(0, qubit_num), 2)
        shuffle(lst)
        i = lst[0]
        j = lst[1]
        CU3((uniform(0, 3.14), uniform(0, 3.14), uniform(0, 3.14))) | circuit([i, j])

    out_circuit_to_file(qubit_num, "cu3.txt", circuit)
    circuit.clear()

    for i in range(0, qubit_num, 2):
        H | circuit(i)
    for i in range(qubit_num):
        X | circuit(i)
    for _ in range(qubit_num):
        X | circuit(randint(0, qubit_num - 1))

    out_circuit_to_file(qubit_num, "x.txt", circuit)
    circuit.clear()

    for tiny_circuit_qubit_num in range(1, 5):
        tiny_circuit = Circuit(tiny_circuit_qubit_num)

        # H
        for i in range(tiny_circuit_qubit_num):
            H | tiny_circuit(i)
        out_circuit_to_file(tiny_circuit_qubit_num, f"tiny_h_{tiny_circuit_qubit_num}.txt", tiny_circuit)
        tiny_circuit.clear()

        # Diag
        for i in range(tiny_circuit_qubit_num):
            H | tiny_circuit(i)
        for i in range(tiny_circuit_qubit_num):
            for _ in range(4):
                Rz(uniform(0, 3.14)) | tiny_circuit(i)
        out_circuit_to_file(tiny_circuit_qubit_num, f"tiny_diag_{tiny_circuit_qubit_num}.txt", tiny_circuit)
        tiny_circuit.clear()

        # X
        for i in range(0, tiny_circuit_qubit_num, 2):
            H | tiny_circuit(i)
        for i in range(tiny_circuit_qubit_num):
            X | tiny_circuit(i)
        for _ in range(tiny_circuit_qubit_num):
            X | tiny_circuit(randint(0, tiny_circuit_qubit_num - 1))
        out_circuit_to_file(tiny_circuit_qubit_num, f"tiny_x_{tiny_circuit_qubit_num}.txt", tiny_circuit)
        tiny_circuit.clear()


if __name__ == '__main__':
    main()
