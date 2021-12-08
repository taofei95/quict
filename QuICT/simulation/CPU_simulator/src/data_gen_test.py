#!/usr/bin/env python3

from random import randint, sample, shuffle, uniform
from scipy.stats import unitary_group
import numpy as np
from QuICT.algorithm import Amplitude
from QuICT.core import *


def format_cpx(n):
    return '{0}{1}{2}j'.format(n.real, '+-'[int(n.imag < 0)], abs(n.imag))


diag_1_gates = {GATE_ID["Rz"]}
diag_2_gates = {GATE_ID["Rzz"]}
ctrl_diag_gates = {GATE_ID["Crz"], GATE_ID["CU1"]}

ctrl_unitary_gate = {GATE_ID["CU3"]}


def out_circuit_to_file(qubit_num: int, f_name: str, circuit: Circuit, result=None):
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
            elif gate.type() in diag_2_gates:  # all 2-bit diagonal gates
                print(f"diag_2 {gate.affectArgs[0]} {gate.affectArgs[1]} "
                      f"{format_cpx(gate.compute_matrix[0, 0])} {format_cpx(gate.compute_matrix[1, 1])} "
                      f"{format_cpx(gate.compute_matrix[2, 2])} {format_cpx(gate.compute_matrix[3, 3])}", file=f)
            elif gate.type() in ctrl_diag_gates:
                print(f"ctrl_diag {gate.carg} {gate.targ} "
                      f"{format_cpx(gate.compute_matrix[2, 2])} {format_cpx(gate.compute_matrix[3, 3])}", file=f)
            elif gate.type() in ctrl_unitary_gate:
                print(f"ctrl_unitary {gate.carg} {gate.targ} ", end="", file=f)
                for i in range(2, 4):
                    for j in range(2, 4):
                        print(f"{format_cpx(gate.compute_matrix[i, j])} ", end="", file=f)
                print("", file=f)
            elif gate.type() == GATE_ID['Measure']:
                print(f"measure {gate.targ}", file=f)
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

        if result is None:
            res = Amplitude.run(circuit)
        else:
            res = result

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
        theta = random.random() * 2 * np.pi
        return random.choice([Rx, Ry])(theta)
    elif qubit_num == 2:
        gate = random.choice([
            Rxx(uniform(0, 2 * np.pi)),
            Ryy(uniform(0, 2 * np.pi)),
            FSim([uniform(0, 2 * np.pi), uniform(0, 2 * np.pi)])
        ])
        # print(gate.qasm_name, gate.pargs)
        return gate

def measure_test():
    # Measure gate test
    qubit_num = 5
    gate_num = 10
    measure_num = 3
    n_run = 10000

    gates = []
    for i in range(qubit_num):
        gates.append([H, i])
    for i in range(gate_num):
        gates.append([manual_rand_unitary_gate(2), sample(range(0, qubit_num), 2)])
    for i in range(measure_num):
        gates.append([Measure, randint(0, qubit_num - 1)])

    res = np.zeros(1 << qubit_num, dtype=complex)
    for _ in range(n_run):
        circuit = Circuit(qubit_num)
        for gate, targ in gates:
            gate | circuit(targ)
        res += Amplitude.run(circuit)

    out_circuit_to_file(qubit_num,
                        "./test_data/measure.txt",
                        circuit,
                        result=res / n_run)

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

    out_circuit_to_file(qubit_num, "./test_data/u.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)

    for _ in range(30):
        Rz(uniform(0, 3.14)) | circuit(randint(0, qubit_num - 1))
    out_circuit_to_file(qubit_num, "./test_data/diag.txt", circuit)
    circuit.clear()

    QFT(qubit_num).build_gate() | circuit
    out_circuit_to_file(qubit_num, "./test_data/qft.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)

    out_circuit_to_file(qubit_num, "./test_data/h.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)
    for _ in range(qubit_num * 40):
        lst = sample(range(0, qubit_num), 2)
        shuffle(lst)
        i = lst[0]
        j = lst[1]
        CRz(uniform(0, 3.14)) | circuit([i, j])

    out_circuit_to_file(qubit_num, "./test_data/crz.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)
    for _ in range(qubit_num * 40):
        lst = sample(range(0, qubit_num), 2)
        shuffle(lst)
        i = lst[0]
        j = lst[1]
        CU3((uniform(0, 3.14), uniform(0, 3.14), uniform(0, 3.14))) | circuit([i, j])

    out_circuit_to_file(qubit_num, "./test_data/cu.txt", circuit)
    circuit.clear()

    for i in range(0, qubit_num, 2):
        H | circuit(i)
    for i in range(qubit_num):
        X | circuit(i)
    for _ in range(qubit_num):
        X | circuit(randint(0, qubit_num - 1))

    out_circuit_to_file(qubit_num, "./test_data/x.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)

    for _ in range(qubit_num * 40):
        lst = sample(range(0, qubit_num), 2)
        shuffle(lst)
        i = lst[0]
        j = lst[1]
        Rzz(uniform(0, 3.14)) | circuit([i, j])

    out_circuit_to_file(qubit_num, "./test_data/diag_2.txt", circuit)
    circuit.clear()

    del circuit

    for tiny_circuit_qubit_num in range(1, 5):
        tiny_circuit = Circuit(tiny_circuit_qubit_num)

        # H
        for i in range(tiny_circuit_qubit_num):
            H | tiny_circuit(i)
        out_circuit_to_file(tiny_circuit_qubit_num, f"./test_data/tiny_h_{tiny_circuit_qubit_num}.txt", tiny_circuit)
        tiny_circuit.clear()

        # Diag
        for i in range(tiny_circuit_qubit_num):
            H | tiny_circuit(i)
        for i in range(tiny_circuit_qubit_num):
            for _ in range(4):
                Rz(uniform(0, 3.14)) | tiny_circuit(i)
        out_circuit_to_file(tiny_circuit_qubit_num, f"./test_data/tiny_diag_{tiny_circuit_qubit_num}.txt", tiny_circuit)
        tiny_circuit.clear()

        # X
        for i in range(0, tiny_circuit_qubit_num, 2):
            H | tiny_circuit(i)
        for i in range(tiny_circuit_qubit_num):
            X | tiny_circuit(i)
        for _ in range(tiny_circuit_qubit_num):
            X | tiny_circuit(randint(0, tiny_circuit_qubit_num - 1))
        out_circuit_to_file(tiny_circuit_qubit_num, f"./test_data/tiny_x_{tiny_circuit_qubit_num}.txt", tiny_circuit)
        tiny_circuit.clear()

        # Ctrl diag
        if tiny_circuit_qubit_num > 1:
            for i in range(tiny_circuit_qubit_num):
                H | tiny_circuit(i)
            for _ in range(15):
                lst = sample(range(0, tiny_circuit_qubit_num), 2)
                shuffle(lst)
                i = lst[0]
                j = lst[1]
                CRz(uniform(0, 3.14)) | tiny_circuit([i, j])
            out_circuit_to_file(tiny_circuit_qubit_num, f"./test_data/tiny_ctrl_diag_{tiny_circuit_qubit_num}.txt",
                                tiny_circuit)
            tiny_circuit.clear()

        # Unitary
        for i in range(tiny_circuit_qubit_num):
            H | tiny_circuit(i)
        for _ in range(15):
            n = randint(1, 2)
            n = min(n, tiny_circuit_qubit_num)
            gate = manual_rand_unitary_gate(n)
            if n == 1:
                gate | tiny_circuit(randint(0, tiny_circuit_qubit_num - 1))
            else:
                gate | tiny_circuit(sample(range(0, tiny_circuit_qubit_num), 2))
        out_circuit_to_file(tiny_circuit_qubit_num, f"./test_data/tiny_unitary_{tiny_circuit_qubit_num}.txt",
                            tiny_circuit)
        tiny_circuit.clear()

        # Ctrl unitary
        if tiny_circuit_qubit_num > 1:
            for i in range(tiny_circuit_qubit_num):
                H | tiny_circuit(i)
            for _ in range(10):
                lst = sample(range(0, tiny_circuit_qubit_num), 2)
                shuffle(lst)
                i = lst[0]
                j = lst[1]
                CU3((uniform(0, 3.14), uniform(0, 3.14), uniform(0, 3.14))) | tiny_circuit([i, j])
            out_circuit_to_file(tiny_circuit_qubit_num,
                                f"./test_data/tiny_ctrl_unitary_{tiny_circuit_qubit_num}.txt",
                                tiny_circuit)
            tiny_circuit.clear()

    measure_test()

if __name__ == '__main__':
    main()