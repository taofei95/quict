#!/usr/bin/env python3

from random import randint, uniform

from QuICT.core import *
from QuICT.algorithm import Amplitude


def out_circuit_to_file(f_name: str, circuit: Circuit):
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

        # res = Amplitude.run(circuit)
        # for val in res:
        #     if abs(val - 0) <= 1e-8:
        #         print("0+0j", file=f)
        #     else:
        #         print(str(val)[1:-1], file=f)


qubit_num = 27

circuit = Circuit(qubit_num)

QFT.build_gate(qubit_num) | circuit

out_circuit_to_file("qft_perf.txt", circuit)

circuit.clear()

for i in range(qubit_num):
    H | circuit(i)

out_circuit_to_file("h_perf.txt", circuit)


