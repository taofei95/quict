#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/18 7:19 上午
# @Author  : Han Yu
# @File    : qasm_checker.py


#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/3/23 4:04 下午
# @Author  : Han Yu
# @File    : QASM_checker.py

from QuICT.tools.interface import OPENQASMInterface

# load qasm
qasm = OPENQASMInterface.load_file("../qasm/pea_3_pi_8.qasm")
if qasm.valid_circuit:
    # generate circuit
    circuit = qasm.circuit
    circuit.print_infomation()

    new_qasm = OPENQASMInterface.load_circuit(circuit)
    new_qasm.output_qasm("test.qasm")
else:
    print("Invalid format!")
