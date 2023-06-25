#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/18 7:19 上午
# @Author  : Han Yu
# @File    : qasm_checker.py
import os

from QuICT.tools.interface import OPENQASMInterface


# load qasm
file_path = os.path.join(os.path.dirname(__file__), "test.qasm")
qasm = OPENQASMInterface.load_file(file_path)
if qasm.valid_circuit:
    # generate circuit
    circuit = qasm.circuit
    print(circuit.qasm())
else:
    print("Invalid format!")

new_qasm = OPENQASMInterface.load_circuit(circuit)
new_qasm.output_qasm()
