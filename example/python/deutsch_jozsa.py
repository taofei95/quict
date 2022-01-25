#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/23 10:14 上午
# @Author  : Han Yu
# @File    : Deutsch_Jozsa.py

from QuICT.core import Circuit
from QuICT.core.gate import H, X, Measure, PermFx
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator


def run_deutsch_jozsa(n, oracle):
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
    X | circuit(ancilla)
    H | circuit

    # Apply oracle U_f which flips the phase of every state |x> with f(x) = 1
    oracle | circuit

    # Apply H
    for q in qreg:
        H | circuit(q)
    # Measure
    Measure | circuit

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
    oracle = PermFx(test_number, test)
    run_deutsch_jozsa(test_number, oracle)
