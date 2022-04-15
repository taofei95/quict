from QuICT.core.circuit import Circuit
from QuICT.core.gate import *

class TemplateLib:

    def __init__(self, qubit_num, size, depth):
        self.qubit_num = qubit_num
        self.size = size
        self.depth = depth
        self.template = []

        if self.size >= 4 and self.depth >= 4:
            circuit = Circuit(self.qubit_num)
            S | circuit(0)
            S | circuit(0)
            S | circuit(0)
            S | circuit(0)
            self.template.append(circuit)

        if self.size >= 2 and self.depth >= 2:
            circuit = Circuit(self.qubit_num)
            H | circuit(0)
            H | circuit(0)
            self.template.append(circuit)

        if self.qubit_num >= 2 and self.size >= 2 and self.depth >= 2:
            circuit = Circuit(self.qubit_num)
            CX | circuit([0,1])
            CX | circuit([0,1])
            self.template.append(circuit)

        if self.qubit_num >= 2 and self.size >= 6 and self.depth >= 4:
            circuit = Circuit(self.qubit_num)
            H | circuit(0)
            H | circuit(1)
            CX | circuit([0, 1])
            H | circuit(0)
            H | circuit(1)
            CX | circuit([1, 0])
            self.template.append(circuit)

        if self.qubit_num >= 2 and self.size >= 6 and self.depth >= 5:
            circuit = Circuit(self.qubit_num)
            H | circuit(0)
            CX | circuit([0, 1])
            H | circuit(1)
            H | circuit(0)
            CX | circuit([1, 0])
            H | circuit(1)
            self.template.append(circuit)

        if self.qubit_num >= 2 and self.size >= 6 and self.depth >= 5:
            circuit = Circuit(self.qubit_num)
            H | circuit(0)
            CX | circuit([1, 0])
            H | circuit(1)
            H | circuit(0)
            CX | circuit([0, 1])
            H | circuit(1)
            self.template.append(circuit)

        if self.qubit_num >= 2 and self.size >= 6 and self.depth >= 4:
            circuit = Circuit(self.qubit_num)
            CX | circuit([0, 1])
            H | circuit(0)
            H | circuit(1)
            CX | circuit([1, 0])
            H | circuit(0)
            H | circuit(1)
            self.template.append(circuit)

        if self.qubit_num >= 3 and self.size >= 5 and self.depth >= 5:
            circuit = Circuit(self.qubit_num)
            CX | circuit([0, 1])
            CX | circuit([0, 2])
            CX | circuit([1, 2])
            CX | circuit([0, 1])
            CX | circuit([1, 2])
            self.template.append(circuit)

        if self.qubit_num >= 2 and self.size >= 6 and self.depth >= 6:
            circuit = Circuit(self.qubit_num)
            CX | circuit([0, 1])
            CX | circuit([1, 0])
            CX | circuit([0, 1])
            CX | circuit([1, 0])
            CX | circuit([0, 1])
            CX | circuit([1, 0])
            self.template.append(circuit)

        if self.qubit_num >= 3 and self.size >= 6 and self.depth >= 6:
            circuit = Circuit(self.qubit_num)
            CX | circuit([0, 1])
            CX | circuit([1, 0])
            CX | circuit([0, 2])
            CX | circuit([1, 0])
            CX | circuit([0, 1])
            CX | circuit([1, 2])
            self.template.append(circuit)

        if self.qubit_num >= 3 and self.size >= 6 and self.depth >= 6:
            circuit = Circuit(self.qubit_num)
            CX | circuit([0, 1])
            CX | circuit([1, 0])
            CX | circuit([2, 1])
            CX | circuit([1, 0])
            CX | circuit([0, 1])
            CX | circuit([2, 0])
            self.template.append(circuit)

if __name__ == '__main__':
    qubit_num, size, depth = eval(input('请输入参数（比特数，规模，深度）：'))
    template = TemplateLib(qubit_num, size, depth).template
    label = 1
    for item_circuit in template:
        print(item_circuit.draw('matp', str(label)))
        label += 1
