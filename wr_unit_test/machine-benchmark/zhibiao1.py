from curses.ascii import isdigit
import os
import re
import numpy as np
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.optimization import *
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

single_typelist = [GateType.h, GateType.rx, GateType.ry, GateType.rz] 
double_typelist = [GateType.cx]
len_s, len_d = len(single_typelist), len(double_typelist)
prob = [0.8 / len_s] * len_s + [0.2 / len_d] * len_d

nq = 5
circuit = Circuit(nq)
circuit.random_append(rand_size=100, typelist=single_typelist + double_typelist, probabilities=prob, random_params=True)

ng = circuit.size()
d = circuit.depth()

P = (ng/d -1)/(nq - 1)
print(P)

