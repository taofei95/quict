from QuICT.tools.interface import OPENQASMInterface
from QuICT.qcda.optimization.template_optimization.template_searching.lib import TemplateLib
import os

class CircuitLib:

    """ Circuit Library

    get_circuit(name, *args)
    
    name: string, circuit type name (template/random/algorithm/experiment}
    
    *args:
    
        template:   args[0]: int, number of qubits
                    args[1]: int, size of qubits
                    args[2]: int, depth of qubits

        random:     args[0]: string, random circuit type (ctrl_diag/ctrl_unitary/diag/qft/single_bit/unitary)
                    args[1]: stirng (small/medium/large) or int, size type

        algorithm:  args[0]: string, algorithm circuit type (QFT/Grover/Supermacy)
                    args[1]: int, size type

        Experiment: args[0]: string, experiment circuit type (Mapping/Adder)
                    args[1]: for Mapping: string, circuit name
                             for Adder:   int, circuit size

"""

    def load_qasm(self, filename):
        qasm = OPENQASMInterface.load_file(filename)
        if not qasm.valid_circuit:
            raise QasmError("Missing input file")
        return qasm.circuit

    def load_all(self, filename):
        circuit_all = []
        for root, dirs, files in os.walk(filename):
            for file in files:
                circuit_all.append(self.load_qasm(filename + '/' + file))
        return circuit_all

    def get_circuit(self, name="template", *args):
        
        para = args
        circuit_list = []
        filename = os.path.dirname(os.path.abspath(__file__)) + '/circuit_qasm/' + name
        
        if name == "template":
            for bit_num in range(para[0] + 1):
                for size in range(para[1] + 1):
                    for depth in range(para[2] + 1):
                        part_name = str(bit_num) + '_' + str(size) + '_' + str(depth)
                        for root, dirs, files in os.walk(filename):
                            for file in files:
                                if file.find(part_name) != -1:
                                    circuit_list.append(self.load_qasm(filename + '/' + file))
        elif name == "random":
            filename = filename + '/' + para[0]
            list_dict = {"small":[1,2,3,4,5], "middle":[13,14,15,16,17,18,19], "large":[30,35,40,45,50]}
            if para[1] in list_dict:
                random_list = list_dict[para[1]]
            else:
                random_list = [para[1]]
            for list_name in random_list:
                filename_temp = filename + '/' + str(list_name) + ".qasm"
                circuit  = self.load_qasm(filename_temp)
                if circuit != None:
                    circuit_list.append(circuit)

        elif name == "algorithm":
            filename = filename + '/' + para[0]
            list_dict = {"QFT":"/qft_", "Grover":"/grover_", "Supermacy":"/"}
            if para[0] in list_dict:
                filename = filename + list_dict[para[0]] + str(para[1]) + ".qasm"
                circuit  = self.load_qasm(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            else:
                filename = ""

        elif name == "experiment":
            filename = filename + '/' + para[0]
            if para[0] == "Mapping":
                filename = filename + '/' + para[1]
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
