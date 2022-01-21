#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/23 10:14 上午
# @Author  : Han Yu
# @File    : Deutsch_Jozsa.py

from QuICT.core import Circuit
from QuICT.core.gate import H, X, Measure, PermFx
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator


def deutsch_jozsa_main_oracle(f, n, circuit):
    PermFx(n, f) | circuit


def run_deutsch_jozsa(f, n, oracle):
    """ an oracle, use Deutsch_Jozsa to decide whether f is balanced

    f(list): the function to be decided
    n(int): the input bit
    oracle(function): oracle function
    """

    # Determine number of qreg
    circuit = Circuit(n + 1)

    # start the eng and allocate qubits
    qreg = circuit[[i for i in range(n)]]
    ancilla = circuit[n]

    # Start with qreg in equal superposition and ancilla in |->
    H | circuit(qreg)
    X | circuit(ancilla)
    H | circuit(ancilla)

    # Apply oracle U_f which flips the phase of every state |x> with f(x) = 1
    oracle(f, n, circuit)

    # Apply H
    H | qreg
    # Measure
    Measure | qreg
    Measure | ancilla

    simulator = ConstantStateVectorSimulator()
    _ = simulator.run(circuit)

    y = int(qreg)

    if y == 0:
        print('Function is constant. y={}'.format(y))
    else:
        print('Function is balanced. y={}'.format(y))


if __name__ == '__main__':
    test_number = 5
    test = [i for i in range(1, 2 ** test_number, 2)]
    run_deutsch_jozsa(test, test_number, deutsch_jozsa_main_oracle)
