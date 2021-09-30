#!/usr/bin/env python3

from random import choices, randint, sample, shuffle, uniform
from random import choice
from scipy.stats import unitary_group

from QuICT.core import *
from QuICT.algorithm import Amplitude
from math import sqrt, pi


def format_cpx(n):
    return '{0}{1}{2}j'.format(n.real, '+-'[int(n.imag < 0)], abs(n.imag))


def out_circuit_to_file(qubit_num: int, f_name: str, circuit: Circuit):
    with open(f_name, 'w') as f:
        print(qubit_num, file=f)

        for gate in circuit.gates:
            gate: BasicGate
            if gate.type() == GATE_ID['H']:
                print(f"h {gate.targ}", file=f)
            elif gate.type() == GATE_ID['Crz']:
                print(f"crz {gate.carg} {gate.targ} {gate.parg}", file=f)
            elif gate.type() == GATE_ID['X']:
                print(f"x {gate.targ}", file=f)
            elif gate.type() == GATE_ID['CU3']:
                print(f"cu3 {gate.carg} {gate.targ} {gate.pargs[0]} {gate.pargs[1]} {gate.pargs[2]}", file=f)
            else:
                if gate.compute_matrix.shape[0] == 2:
                    print(f'u1 {gate.targ}', file=f, end=' ')
                elif gate.compute_matrix.shape[1] == 4:
                    print(f'u2 {gate.affectArgs[0]} {gate.affectArgs[1]}', file=f, end=' ')
                for c in gate.compute_matrix.flatten():
                    print(format_cpx(c), file=f, end=' ')
                print(file=f)

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
            gate | circuit(randint(0, qubit_num-1))
        else:
            gate | circuit(sample(range(0, qubit_num), 2))

    out_circuit_to_file(qubit_num, "u.txt", circuit)
    circuit.clear()

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

    for i in range(qubit_num):
        H | circuit(i)
    for i in range(qubit_num):
        X | circuit(i)
    for _ in range(qubit_num):
        X | circuit(randint(0, qubit_num - 1))

    out_circuit_to_file(qubit_num, "x.txt", circuit)
    circuit.clear()


if __name__ == '__main__':
    main()
