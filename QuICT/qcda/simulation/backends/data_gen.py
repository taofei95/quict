#!/usr/bin/env python3

from random import randint, uniform

from QuICT.core import *
from QuICT.algorithm import Amplitude


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

    QFT.build_gate(qubit_num) | circuit
    out_circuit_to_file(qubit_num, "qft.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)
    for i in range(200):
        H | circuit(randint(0, qubit_num - 3))
    H | circuit(qubit_num - 1)
    H | circuit(qubit_num - 2)
    H | circuit(qubit_num - 3)

    out_circuit_to_file(qubit_num, "h.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        H | circuit(i)
    for i in range(1, qubit_num):
        CRz(uniform(0, 3.14)) | circuit([0, i])

    out_circuit_to_file(qubit_num, "crz.txt", circuit)
    circuit.clear()

    for i in range(qubit_num):
        X | circuit(i)
    for i in range(200):
        X | circuit(randint(0, qubit_num - 3))
    X | circuit(qubit_num - 1)
    X | circuit(qubit_num - 2)
    X | circuit(qubit_num - 3)

    out_circuit_to_file(qubit_num, "x.txt", circuit)
    circuit.clear()


if __name__ == '__main__':
    main()
