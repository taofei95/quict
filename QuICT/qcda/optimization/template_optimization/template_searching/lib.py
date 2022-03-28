from QuICT.core.circuit import Circuit
from QuICT.core.gate import *

class GetTemplate:

    def __init__(self, qubit_num, size, depth):
        self.qubit_num = qubit_num
        self.size = size
        self.depth = depth

    def get_template(self):

        template_list = []

        if self.qubit_num >= 2 and self.size >= 6 and self.depth >= 4:
            circuit = Circuit(self.qubit_num)
            H | circuit(0)
            H | circuit(1)
            CX | circuit([0, 1])
            H | circuit(0)
            H | circuit(1)
            CX | circuit([1, 0])
            template_list.append(circuit)

        if self.qubit_num >= 2 and self.size >= 6 and self.depth >= 5:
            circuit = Circuit(self.qubit_num)
            H | circuit(0)
            CX | circuit([0, 1])
            H | circuit(1)
            H | circuit(0)
            CX | circuit([1, 0])
            H | circuit(1)
            template_list.append(circuit)

        if self.qubit_num >= 2 and self.size >= 6 and self.depth >= 5:
            circuit = Circuit(self.qubit_num)
            H | circuit(0)
            CX | circuit([1, 0])
            H | circuit(1)
            H | circuit(0)
            CX | circuit([0, 1])
            H | circuit(1)
            template_list.append(circuit)

        if self.qubit_num >= 2 and self.size >= 6 and self.depth >= 4:
            circuit = Circuit(self.qubit_num)
            CX | circuit([0, 1])
            H | circuit(0)
            H | circuit(1)
            CX | circuit([1, 0])
            H | circuit(0)
            H | circuit(1)
            template_list.append(circuit)

        if self.qubit_num >= 3 and self.size >= 5 and self.depth >= 5:
            circuit = Circuit(self.qubit_num)
            CX | circuit([0, 1])
            CX | circuit([0, 2])
            CX | circuit([1, 2])
            CX | circuit([0, 1])
            CX | circuit([1, 2])
            template_list.append(circuit)

        if self.qubit_num >= 2 and self.size >= 6 and self.depth >= 6:
            circuit = Circuit(self.qubit_num)
            CX | circuit([0, 1])
            CX | circuit([1, 0])
            CX | circuit([0, 1])
            CX | circuit([1, 0])
            CX | circuit([0, 1])
            CX | circuit([1, 0])
            template_list.append(circuit)

        if self.qubit_num >= 3 and self.size >= 6 and self.depth >= 6:
            circuit = Circuit(self.qubit_num)
            CX | circuit([0, 1])
            CX | circuit([1, 0])
            CX | circuit([0, 2])
            CX | circuit([1, 0])
            CX | circuit([0, 1])
            CX | circuit([1, 2])
            template_list.append(circuit)

        if self.qubit_num >= 3 and self.size >= 6 and self.depth >= 6:
            circuit = Circuit(self.qubit_num)
            CX | circuit([0, 1])
            CX | circuit([1, 0])
            CX | circuit([2, 1])
            CX | circuit([1, 0])
            CX | circuit([0, 1])
            CX | circuit([2, 0])
            template_list.append(circuit)

        return template_list

qubit_num, size, depth = eval(input('请输入参数（比特数，规模，深度）：'))
template = GetTemplate(qubit_num, size, depth)
list_circuit = template.get_template()
label = 1
for item_circuit in list_circuit:
    print(item_circuit.draw('matp', str(label)))
    label += 1