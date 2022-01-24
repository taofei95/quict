#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/18 7:19 上午
# @Author  : Han Yu
# @File    : qasm_checker.py

from QuICT.tools.interface import OPENQASMInterface
from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator
from QuICT.core import *


# load qasm
qasm = OPENQASMInterface.load_file("../qasm/test.qasm")
if qasm.valid_circuit:
    # generate circuit
    circuit = qasm.circuit
    print(circuit.qasm())

    simulator = ConstantStateVectorSimulator()
    state = simulator.run(circuit)

    print(state)

    new_qasm = OPENQASMInterface.load_circuit(circuit)
    new_qasm.output_qasm("test.qasm")
else:
    print("Invalid format!")
