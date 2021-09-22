#!/usr/bin/env python3

from random import choices, randint, sample, shuffle, uniform
from random import choice

from QuICT.core import *
from QuICT.algorithm import Amplitude
from math import sqrt


def out_unitary_circuit_to_file(qubit_num: int, f_name: str, circuit: Circuit):
    with open(f_name, 'w') as f:
        print(qubit_num, file=f)

        for gate in circuit.gates:
            gate: BasicGate
            if gate.type() == GATE_ID['H']:
                val_pos = str(complex(1/sqrt(2), 0))[1:-1]
                val_neg = str(complex(-1/sqrt(2), 0))[1:-1]
                print(
                    f"u1 {gate.targ} {val_pos} {val_pos} {val_pos} {val_neg}", file=f)
            elif gate.type() == GATE_ID['X']:
                print(f"u1 {gate.targ} 0+0j 1+0j 1+0j 0+0j", file=f)
            elif gate.type() == GATE_ID['S']:
                print(f"u1 {gate.targ} 1+0j 0+0j 0+0j 0+1j", file=f)
            elif gate.type() == GATE_ID['T']:
                val = str(complex(1/sqrt(2), 1/sqrt(2)))[1:-1]
                print(f"u1 {gate.targ} 1+0j 0+0j 0+0j {val}", file=f)

        print("__TERM__", file=f)

        res = Amplitude.run(circuit)
        for val in res:
            if abs(val - 0) <= 1e-8:
                print("0+0j", file=f)
            else:
                print("%s%s%sj" % (val.real, '+' if val.imag >=
                      0 else '-', abs(val.imag)), file=f)
                # opt = str(val)[1:-1]
                # print(opt, file=f)


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

        print("__TERM__", file=f)

        res = Amplitude.run(circuit)
        for val in res:
            if abs(val - 0) <= 1e-8:
                print("0+0j", file=f)
            else:
                print(str(val)[1:-1], file=f)


def main():
    qubit_num = 18
    circuit = Circuit(qubit_num)

    QFT(qubit_num).build_gate() | circuit
    out_circuit_to_file(qubit_num, "qft.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)

    out_circuit_to_file(qubit_num, "h.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)
    for _ in range(qubit_num*30):
        lst = sample(range(0, qubit_num),2)
        shuffle(lst)
        i = lst[0]
        j = lst[1]
        CRz(uniform(0, 3.14)) | circuit([i,j])

    out_circuit_to_file(qubit_num, "crz.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)
    for i in range(qubit_num):
        X | circuit(i)
    for _ in range(qubit_num):
        X | circuit(randint(0, qubit_num - 1))

    out_circuit_to_file(qubit_num, "x.txt", circuit)
    circuit.clear()

    gates = [H, X, S, T]
    for i in range(200):
        random.choice(gates) | circuit(randint(0, qubit_num - 1))

    # H | circuit(0)
    # H | circuit(0)
    out_unitary_circuit_to_file(qubit_num, "u1.txt", circuit)
    circuit.clear()


if __name__ == '__main__':
    main()
