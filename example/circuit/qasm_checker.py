#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/18 7:19 上午
# @Author  : Han Yu
# @File    : qasm_checker.py

from QuICT.tools.interface import OPENQASMInterface


# load qasm
qasm = OPENQASMInterface.load_file("./test.qasm")
if qasm.valid_circuit:
    # generate circuit
    circuit = qasm.circuit
    print(circuit.qasm())    
else:
    print("Invalid format!")

new_qasm = OPENQASMInterface.load_circuit(circuit)
new_qasm.output_qasm()
