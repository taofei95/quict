#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/5/3 4:04 下午
# @Author  : Han Yu
# @File    : circuit_draw
import os

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.tools.interface import OPENQASMInterface


def circuit_build():
    qubits = 4     # qubits number
    cir = Circuit(qubits)
    qureg = cir.qubits

    # Add Gate into Circuit
    H | cir(0)
    X | cir     # Add X Gates to all qubits
    U1(np.pi / 2) | cir(1)
    CX | cir([1, 2])
    Rzz(np.pi / 2) | cir([1, 2])
    Barrier | cir(2)
    iSwap_dagger | cir([0, 1])

    # Add Gate into Circuit Through Qureg
    S | cir(qureg[0])
    CY | cir(qureg[1, 3])
    CSwap | cir(qureg[3, 1, 0])

    # Add pre-defined Gate into Circuit
    c1 = Ryy & [1, 3]
    c1(np.pi / 4) | cir

    # Add Unitary Gate into Circuit
    matrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, -1]
    ], dtype=np.complex128)
    cgate_complex4 = Unitary(matrix, MatrixType.identity)
    u_gate = cgate_complex4.build_gate()
    u_gate | cir

    # Add CompositeGate into Circuit
    cgate = CompositeGate()
    CCRz(1) | cgate([0, 1, 2])
    QFT(3) | cgate
    cgate | cir([1, 2, 3])

    print(cir.qasm())
    cir.draw(filename="circuit_build")


def random_build():
    circuit = Circuit(5)
    circuit.random_append(30)
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
    circuit_build()
