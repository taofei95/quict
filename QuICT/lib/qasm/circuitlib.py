from QuICT.tools.interface import OPENQASMInterface
from QuICT.qcda.optimization.template_optimization.template_searching.lib import TemplateLib

class CircuitLib:

    def __init__(self, name="template", *args):
        self.name = name
        self.para = args

    def load_qasm(self, filename):
        fileaddress = "./circuit_qasm/"+filename
        qasm = OPENQASMInterface.load_file(fileaddress)
        if qasm.valid_circuit:
            circuit = qasm.circuit
            # print(circuit.qasm())
        else:
            circuit = None
            print("Invalid format!")
        return circuit

    def get_circuit(self):
        circuit_list = []
        if self.name == "template":
            bit_num = self.para[0]
            size = self.para[1]
            depth = self.para[2]
            circuit_list = TemplateLib(bit_num,size,depth).template_list()
        elif self.name == "random":
            filename = self.name
            filename = filename + '_' + self.para[0]
            if self.para[0] == "ctrl_diag":
                filename = filename + '_' + str(self.para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            elif self.para[0] == "ctrl_unitary":
                filename = filename + '_' + str(self.para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            elif self.para[0]== "diag":
                filename = filename + '_' + str(self.para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            elif self.para[0] == "qft":
                filename = filename + '_' + str(self.para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            elif self.para[0]== "single_bit":
                filename = filename + '_' + str(self.para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            elif self.para[0]== "unitary":
                filename = filename + '_' + str(self.para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            else:
                filename = ""

        elif self.name == "algorithm":
            filename = self.name
            filename = filename + '_' + self.para[0]
            if self.para[0] == "QFT":
                filename = filename + '_' + str(self.para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            elif self.para[0] == "Adder":
                filename = filename + '_' + str(self.para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            elif self.para[0]== "QFTAdd":
                filename = filename + '_' + str(self.para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            else:
                filename = ""
        elif self.name == "experiment":
            filename = self.para[0]
            filename = filename + '_' + self.para[0]
            if self.para[0] == "mapping":
                filename = filename + '_' + self.para[1] + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            else:
                filename = ""
        else:
            filename = ""
        return circuit_list

if __name__ == '__main__':
    circuit_list = CircuitLib("random","qft",14).get_circuit()
    print(len(circuit_list))
    for circuit in circuit_list:
        print(circuit.qasm())
