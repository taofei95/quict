from QuICT.tools.interface import OPENQASMInterface
from QuICT.qcda.optimization.template_optimization.template_searching.lib import TemplateLib
import os

""" Circuit Library

    get_circuit(name, *args)
    
    name: template, random, algorithm, experiment
    
    *args:
    
        template:   qubit_num, 
                    size, 
                    depth
        random:     circuit_type (ctrl_diag/ctrl_unitary/diag/qft/single_bit/unitary),
                    size_type (small/medium/large) or qubit number
        algorithm:  circuit_type (QFT/Grover/Supermacy), 
                    qubit number
        Experiment: circuit_type(Mapping/Adder),
                    qubit number

"""

class CircuitLib:

    def load_qasm(self, filename):
        qasm = OPENQASMInterface.load_file(filename)
        if not qasm.valid_circuit:
            raise QasmError("Missing input file")
        return qasm.circuit

    def load_all(self, filename):
        circuit_all = []
        for root, dirs, files in os.walk(filename):
            for file in files:
                qasm = OPENQASMInterface.load_file(file)
                if not qasm.valid_circuit:
                    raise QasmError("Missing input file")
                circuit_all.append(qasm.circuit)
        return circuit_all

    def get_circuit(self, name="template", *args):
        
        para = args
        circuit_list = []
        filename = "./circuit_qasm/"+name
        
        if name == "template":
            for bit_num in range(para[0]):
                for size in range(para[1]):
                    for depth in range(para[2]):
                        part_name = str(bit_num) + '_' + str(size) + '_' + str(depth)
                        for root, dirs, files in os.walk(filename):
                            for file in files:
                                if file.find(part_name) != -1:
                                    qasm = OPENQASMInterface.load_file(file)
                                    if not qasm.valid_circuit:
                                        raise QasmError("Missing input file")
                                    circuit_list.append(qasm.circuit)
        elif name == "random":
            filename = filename + '/' + para[0]
            if para[0] == "ctrl_diag":
                random_list = []
                if para[1] == 'small':
                    for i in range(1, 6):
                        random_list.append(i)
                elif para[1] == 'middle':
                    for i in range(13, 20):
                        random_list.append(i)
                elif para[1] == 'large':
                    for i in range(30, 55, 5):
                        random_list.append(i)
                else:
                    random_list.append(para[1])
                for list_name in random_list:
                    filename_temp = filename + '/' + str(list_name) + ".qasm"
                    circuit  = self.load_qasm(filename_temp)
                    if circuit != None:
                        circuit_list.append(circuit)
            elif para[0] == "ctrl_unitary":
                random_list = []
                if para[1] == 'small':
                    for i in range(1, 6):
                        random_list.append(i)
                elif para[1] == 'middle':
                    for i in range(13, 20):
                        random_list.append(i)
                elif para[1] == 'large':
                    for i in range(30, 55, 5):
                        random_list.append(i)
                else:
                    random_list.append(para[1])
                for list_name in random_list:
                    filename_temp = filename + '/' + str(list_name) + ".qasm"
                    circuit  = self.load_qasm(filename_temp)
                    if circuit != None:
                        circuit_list.append(circuit)
            elif para[0] == "diag":
                random_list = []
                if para[1] == 'small':
                    for i in range(1, 6):
                        random_list.append(i)
                elif para[1] == 'middle':
                    for i in range(13, 20):
                        random_list.append(i)
                elif para[1] == 'large':
                    for i in range(30, 55, 5):
                        random_list.append(i)
                else:
                    random_list.append(para[1])
                for list_name in random_list:
                    filename_temp = filename + '/' + str(list_name) + ".qasm"
                    circuit  = self.load_qasm(filename_temp)
                    if circuit != None:
                        circuit_list.append(circuit)
            elif para[0] == "qft":
                random_list = []
                if para[1] == 'small':
                    for i in range(1, 6):
                        random_list.append(i)
                elif para[1] == 'middle':
                    for i in range(13, 20):
                        random_list.append(i)
                elif para[1] == 'large':
                    for i in range(30, 55, 5):
                        random_list.append(i)
                else:
                    random_list.append(para[1])
                for list_name in random_list:
                    filename_temp = filename + '/' + str(list_name) + ".qasm"
                    circuit  = self.load_qasm(filename_temp)
                    if circuit != None:
                        circuit_list.append(circuit)
            elif para[0] == "single_bit":
                random_list = []
                if para[1] == 'small':
                    for i in range(1, 6):
                        random_list.append(i)
                elif para[1] == 'middle':
                    for i in range(13, 20):
                        random_list.append(i)
                elif para[1] == 'large':
                    for i in range(30, 55, 5):
                        random_list.append(i)
                else:
                    random_list.append(para[1])
                for list_name in random_list:
                    filename_temp = filename + '/' + str(list_name) + ".qasm"
                    circuit  = self.load_qasm(filename_temp)
                    if circuit != None:
                        circuit_list.append(circuit)
            elif para[0] == "unitary":
                random_list = []
                if para[1] == 'small':
                    for i in range(1, 6):
                        random_list.append(i)
                elif para[1] == 'middle':
                    for i in range(13, 20):
                        random_list.append(i)
                elif para[1] == 'large':
                    for i in range(30, 55, 5):
                        random_list.append(i)
                else:
                    random_list.append(para[1])
                for list_name in random_list:
                    filename_temp = filename + '/' + str(list_name) + ".qasm"
                    circuit  = self.load_qasm(filename_temp)
                    if circuit != None:
                        circuit_list.append(circuit)
            else:
                filename = ""

        elif name == "algorithm":
            filename = filename + '/' + para[0]
            if para[0] == "QFT":
                filename = filename + '/qft_' + str(para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            elif para[0] == "Grover":
                filename = filename + '/grover_' + str(para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            elif para[0]== "Supermacy":
                filename = filename + '/' + str(para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            else:
                filename = ""

        elif name == "experiment":
            filename = filename + '/' + para[0]
            if para[0] == "Mapping":
                circuit  = self.load_all(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            elif para[0] == "Adder":
                filename = filename + '/adder_n' + str(para[1]) + ".qasm"
                circuit = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            else:
                filename = ""
        else:
            filename = ""
        return circuit_list

if __name__ == '__main__':
    circuit_list = CircuitLib().get_circuit("random", "qft", 14)
    print(len(circuit_list))
    for circuit in circuit_list:
        print(circuit.qasm())
