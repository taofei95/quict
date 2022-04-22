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
            if circuit.width() <= qubit_num and circuit.size() <= size and circuit.depth() <= depth:
                self.template.append(circuit)

    def template_list(self):
        return self.template

# if __name__ == '__main__':
#    qubit_num, size, depth = eval(input('请输入参数（比特数，规模，深度）：'))
#    template = TemplateLib(qubit_num, size, depth).template
#    label = 1
#    for item_circuit in template:
#        print(item_circuit.draw('matp', str(label)))
#        label += 1
