#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/18 7:19 上午
# @Author  : Han Yu
# @File    : qasm_checker.py

from QuICT.tools.interface import OPENQASMInterface
from QuICT.qcda.simulation.statevector_simulator import ConstantStateVectorSimulator
from QuICT.core import *

# with open("../qasm/test_2.qasm") as ifile:
#     data = ifile.read()

# qasm = OPENQASMInterface.load_data(data=data)
# load qasm
qasm = OPENQASMInterface.load_file("../qasm/test_2.qasm")
if qasm.valid_circuit:
    # generate circuit
    circuit = qasm.circuit
    circuit.print_information()

    simulator = ConstantStateVectorSimulator(
        circuit=circuit
    )
    state = simulator.run()

    print(state)

    new_qasm = OPENQASMInterface.load_circuit(circuit)
    new_qasm.output_qasm("test.qasm")
else:
    print("Invalid format!")


circuit = Circuit(5)

H | circuit(0)
H | circuit(1)
H | circuit(3)
H | circuit(2)
SW | circuit(0)
SX | circuit(0)
SY | circuit(0)
Phase(0.7853981633974483) | circuit(2)
FSim([0, 1]) | circuit([0, 1])
Rxx(0) | circuit([0, 1])
Ryy(0) | circuit([0, 1])

circuit.print_information()

simulator = ConstantStateVectorSimulator(
    circuit=circuit
)
state = simulator.run()

print(state)
