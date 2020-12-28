"""
Test the template_optimization algorithm with certain benchmarks
"""

import os
import numpy as np

from QuICT.core import * # pylint: disable=unused-wildcard-import
from QuICT.algorithm import SyntheticalUnitary
from QuICT.tools.interface import OPENQASMInterface
from template_optimization import TemplateOptimization
from templates import template_nct_5a_3

def equiv(circuit1, circuit2):
    if circuit1.circuit_width() != circuit2.circuit_width():
        return False
    mat1 = SyntheticalUnitary.run(circuit1, showSU=False)
    mat2 = SyntheticalUnitary.run(circuit2, showSU=False)
    return not np.any(mat1 != mat2)

for root, dirs, files in os.walk('./Arithmetic_and_Toffoli_qasm'):
    for name in files:
        qasm = OPENQASMInterface.load_file(os.path.join(root, name))
        if qasm.valid_circuit:
            circuit = qasm.circuit
            # circuit.print_infomation()
            template_o = TemplateOptimization([template_nct_5a_3()])
            circuit_opt = template_o.run(circuit)
            print(name, len(circuit.gates), len(circuit_opt.gates))
            # equ = equiv(circuit, circuit_opt)
            # if not equ:
            #     circuit.print_infomation()
            #     circuit_opt.print_infomation()
            #     assert 0