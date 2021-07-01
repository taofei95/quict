#!/usr/bin/env python3

from QuICT.core import *
from QuICT.algorithm import Amplitude

qubit_num = 20

circuit = Circuit(qubit_num)

QFT.build_gate(qubit_num) | circuit

with open("qft.txt", 'w') as f:
    print(qubit_num, file=f)

    for gate in circuit.gates:
        gate: BasicGate
        if gate.type() == GATE_ID['H']:
            print(f"h {gate.targ}", file=f)
        elif gate.type() == GATE_ID['Crz']:
            print(f"crz {gate.carg} {gate.targ} {gate.parg}", file=f)

    print("__TERM__", file=f)

    res = Amplitude.run(circuit)
    for val in res:
        print(str(val)[1:-1], file=f)
