from QuICT.core.circuit import Circuit
from QuICT.core.gate import *

template_list = []

circuit = Circuit(1)
S | circuit(0)
S | circuit(0)
S | circuit(0)
S | circuit(0)
template_list.append(circuit)

circuit = Circuit(1)
H | circuit(0)
H | circuit(0)
template_list.append(circuit)

circuit = Circuit(2)
CX | circuit([0, 1])
CX | circuit([0, 1])
template_list.append(circuit)

circuit = Circuit(2)
H | circuit(0)
H | circuit(1)
CX | circuit([0, 1])
H | circuit(0)
H | circuit(1)
CX | circuit([1, 0])
template_list.append(circuit)

circuit = Circuit(2)
H | circuit(0)
CX | circuit([0, 1])
H | circuit(1)
H | circuit(0)
CX | circuit([1, 0])
H | circuit(1)
template_list.append(circuit)

circuit = Circuit(2)
H | circuit(0)
CX | circuit([1, 0])
H | circuit(1)
H | circuit(0)
CX | circuit([0, 1])
H | circuit(1)
template_list.append(circuit)

circuit = Circuit(2)
CX | circuit([0, 1])
H | circuit(0)
H | circuit(1)
CX | circuit([1, 0])
H | circuit(0)
H | circuit(1)
template_list.append(circuit)

circuit = Circuit(3)
CX | circuit([0, 1])
CX | circuit([0, 2])
CX | circuit([1, 2])
CX | circuit([0, 1])
CX | circuit([1, 2])
template_list.append(circuit)

circuit = Circuit(2)
CX | circuit([0, 1])
CX | circuit([1, 0])
CX | circuit([0, 1])
CX | circuit([1, 0])
CX | circuit([0, 1])
CX | circuit([1, 0])
template_list.append(circuit)

circuit = Circuit(3)
CX | circuit([0, 1])
CX | circuit([1, 0])
CX | circuit([0, 2])
CX | circuit([1, 0])
CX | circuit([0, 1])
CX | circuit([1, 2])
template_list.append(circuit)

circuit = Circuit(3)
CX | circuit([0, 1])
CX | circuit([1, 0])
CX | circuit([2, 1])
CX | circuit([1, 0])
CX | circuit([0, 1])
CX | circuit([2, 0])
template_list.append(circuit)


class TemplateLib:

    def __init__(self, qubit_num, size, depth):
        self.template = []

        for circuit in template_list:
            if circuit.width() <= qubit_num:
                if circuit.size() <= size:
                    if circuit.depth() <= depth:
                        self.template.append(circuit)

    def template_list(self):
        return self.template
