"""
Transform the quipper circuit to qasm circuit with pyzx
"""

import os
import pyzx as zx

for root, dirs, files in os.walk('./Arithmetic_and_Toffoli'):
    for name in files:
        circuit = zx.Circuit.from_quipper_file(os.path.join(root, name))
        qasm = open('./Arithmetic_and_Toffoli_qasm/' + name + '.qasm', 'w')
        qasm.write(circuit.to_qasm())
        qasm.close()
