from QuICT.tools.interface import OPENQASMInterface
import os

class CircuitLib:

    """ Circuit Library

    get_circuit(name, *args)
    
    name: template, random, algorithm, experiment
    
    *args:
    
        template:   args[0]: qubit_num, upper bound of qubit number of templates
                    args[1]: size, upper bound of size of templates
                    args[2]: depth, upper bound of depth of templates
        
        random:     args[0]: circuit_type (ctrl_diag/ctrl_unitary/diag/qft/single_bit/unitary), type of random circuits
                    args[1]: size_type (small/medium/large), get a list of random circuits with similar qubit numbers
                             or qubit number, get a single random circuit with the given qubit number

        algorithm:  args[0]: circuit_type (QFT/Grover/Supermacy), type of algorithm circuits
                    args[1]: size_type (small/medium/large), get a list of algorithm circuits with similar qubit numbers (only for QFT/Grover circuits)
                             or qubit number, get a single algorithm circuit with the given qubit number
        Experiment: args[0]: circuit_type(Mapping/Adder), type of experiment circuits
                    args[1]: (for Adder) size_type (small/medium/large), get a list of adder circuits with similar qubit numbers
                                         or qubit number, get a single adder circuit with the given qubit number
                             (for Mapping) mapping_name, get all mapping circuits under the same qubit mapping

    """

    def load_qasm(self, filename):

        # load a qasm file

        qasm = OPENQASMInterface.load_file(filename)
        if not qasm.valid_circuit:
            raise QasmError("Missing input file")
        return qasm.circuit

    def load_all(self, filename):

        # load all qasm files under the same address

        circuit_all = []
        for root, dirs, files in os.walk(filename):
            for file in files:
                circuit_all.append(self.load_qasm(filename + '/' + file))
        return circuit_all

    def get_circuit(self, name="template", *args):
        
        para = args
        circuit_list = []
        filename = os.path.abspath(__file__) + name
        
        if name == "template":

            # get all templates satisfying <=bit_num & <=size & <=depth

            for bit_num in range(para[0] + 1):
                for size in range(para[1] + 1):
                    for depth in range(para[2] + 1):
                        part_name = str(bit_num) + '_' + str(size) + '_' + str(depth)
                        for root, dirs, files in os.walk(filename):
                            for file in files:
                                if file.find(part_name) != -1:
                                    circuit_list.append(self.load_qasm(filename + '/' + file))
                                    
        elif name == "random":

            # get all random circuits

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

            # get all algorithm circuits

            filename = filename + '/' + para[0]
            list_dict = {"small":[11,13,15,17,19], "middle":[51,53,55,57,59], "large":[91,93,95,97,99]}
            list_dict_2 = {"QFT":"/qft_", "Grover":"/grover_", "Supermacy":"/"}
            if para[0] in list_dict_2:
                if para[0] == "Supermacy":
                    filename = filename + list_dict[para[0]] + str(para[1]) + ".qasm"
                    circuit  = self.load_qasm(filename)
                    if circuit != None:
                        circuit_list.append(circuit)
                else:
                    if para[1] in list_dict:
                        algorithm_list = list_dict[para[1]]
                    else:
                        algorithm_list = [para[1]]
                    for list_name in algorithm_list:
                        filename_temp = filename + '/' + str(list_name) + ".qasm"
                        circuit  = self.load_qasm(filename_temp)
                        if circuit != None:
                            circuit_list.append(circuit)
            else:
                filename = ""

        elif name == "experiment":

            # get all experiment circuits

            filename = filename + '/' + para[0]
            if para[0] == "Mapping":
                filename = filename + '/' + para[1]
                circuit  = self.load_all(filename)
                if circuit != None:
                    circuit_list.append(circuit)
            elif para[0] == "Adder":
                filename = filename + '/adder_n'
                list_dict = {"small":[10,13,16,19,22], "middle":[40,43,46,49,52], "large":[70,73,76,79,82]}
                if para[1] in list_dict:
                    adder_list = list_dict[para[1]]
                else:
                    adder_list = [para[1]]
                for list_name in adder_list:
                    filename_temp = filename + str(list_name) + ".qasm"
                    circuit  = self.load_qasm(filename_temp)
                    if circuit != None:
                        circuit_list.append(circuit)
            else:
                filename = ""
        else:
            filename = ""
        return circuit_list

if __name__ == '__main__':
    circuit_list = CircuitLib().get_circuit("random", "qft", "small")
    print(len(circuit_list))
    for circuit in circuit_list:
        print(circuit.qasm())
