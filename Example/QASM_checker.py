#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/3/23 4:04 下午
# @Author  : Han Yu
# @File    : QASM_checker.py

from QuICT.interface import OPENQASMInterface

# 从文件pea_3_pi_8.qasm读取qasm并自动生成电路
qasm = OPENQASMInterface.load_file("pea_3_pi_8.qasm")
if qasm.valid_circuit:
    # 生成电路并打印信息
    circuit = qasm.circuit
    circuit.print_infomation()

    # 再读取电路，生成qasm文件
    new_qasm = OPENQASMInterface.load_circuit(circuit)
    # 将文件输出到test.qasm
    new_qasm.output_qasm("test.qasm")
else:
    print("这个qasm文件无法转化为QuICT中的电路")
