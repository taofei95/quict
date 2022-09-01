from QuICT.tools.interface import OPENQASMInterface
import os

class CircuitLib:

    """ Circuitlib

    A circuit library function. Get circuits with .qasm type

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

        '''
    
        name: "template", "random", "algorithm", "experiment"
    
        *args:
    
                template:   args[0]: qubit_num, upper bound of qubit number of templates
                            args[1]: size, upper bound of size of templates
                            args[2]: depth, upper bound of depth of templates
        
                random:     args[0]: circuit_type ("ctrl_diag"/"ctrl_unitary"/"diag"/"qft"/"single_bit"/"unitary"), type of random circuits
                            args[1]: qubit number (list/int), get a list of circuits (a single circuit) with given qubit numbers

                algorithm:  args[0]: circuit_type ("QFT"/"Grover"/"Supermacy"), type of algorithm circuits
                            args[1]: qubit number (list/int), get a list of circuits (a single circuit) with given qubit numbers
                
                experiment: args[0]: circuit_type("Mapping"/"Adder"), type of experiment circuits
                            args[1]: (for Adder)  qubit number (list/int), get a list of circuits (a single circuit) with given qubit numbers
                                     (for Mapping) mapping_name, get all mapping circuits under the same qubit mapping

        '''
        
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
            if isinstance(para[1], int):
                random_list = [para[1]]
            else:
                random_list = para[1]
            for list_name in random_list:
                filename_temp = filename + '/' + str(list_name) + ".qasm"
                circuit  = self.load_qasm(filename_temp)
                if circuit != None:
                    circuit_list.append(circuit)

        elif name == "algorithm":

            # get all algorithm circuits

            filename = filename + '/' + para[0]
            list_dict_2 = {"QFT":"/qft_", "Grover":"/grover_", "Supermacy":"/"}
            if para[0] in list_dict_2:
                if isinstance(para[1], int):
                        algorithm_list = [[para[1]]
                    else:
                        algorithm_list = para[1]
                if para[0] == "Supermacy":
                    for list_name in algorithm_list:
                        filename = filename + list_dict[para[0]] + str(para[1]) + ".qasm"
                        circuit  = self.load_qasm(filename)
                        if circuit != None:
                            circuit_list.append(circuit)
                else:
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
                if isinstance(para[1], int):
                    adder_list = [para[1]]
                else:
                    adder_list = para[1]
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
