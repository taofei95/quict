#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/3 4:04 下午
# @Author  : Han Yu
# @File    : circuit_draw
import os

from QuICT.core import Circuit
from QuICT.tools.interface import OPENQASMInterface


def circuit_build():
    pass


def random_build():
    circuit = Circuit(5)
    circuit.random_append(50)
    circuit.draw(filename="random")


def supremacy_build():
    qubits = 5     # qubits number
    circuit = Circuit(qubits)
    circuit.supremacy_append(
        repeat=1,   # The repeat times of cycles
        pattern="ABCDCDAB"  # Indicate the circuit cycle
    )

    circuit.draw(filename="supremacy")


def load_circuit_from_qasm():
    # load qasm
    file_path = os.path.join(os.path.dirname(__file__), "test.qasm")
    qasm = OPENQASMInterface.load_file(file_path)
    if qasm.valid_circuit:
        # generate circuit
        circuit = qasm.circuit
        print(circuit.qasm())
    else:
        print("Invalid format!")

    # Save qasm
    new_qasm = OPENQASMInterface.load_circuit(circuit)
    new_qasm.output_qasm("FromCircuit")


if __name__ == "__main__":
    load_circuit_from_qasm()
