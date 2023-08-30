from QuICT.core import Circuit
from QuICT.core.gate import GateType, gate_builder
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
from QuICT.simulation.state_vector import StateVectorSimulator


CIRCUIT_DEBUG_DESCRIBE = """
   ___            ___    ____   _____
  / _ \   _   _  |_ _|  / ___| |_   _|
 | | | | | | | |  | |  | |       | |
 | |_| | | |_| |  | |  | |___    | |
  \__\_\  \___/  |___|  \____|   |_|


Welcome to QuICT Circuit Debug Tools.

Using 'help' to get the documents about QuICT CLI.
"""


class CircuitDebug:
    def __init__(self):
        self._circuit: Circuit = None
        self._simulation = StateVectorSimulator()

    def help_info(self):
        help_str = """
            Load the file of Circuit: load [filepath] \n
            Create an empty Quantum Circuit with n qubits: circuit n \n
            Add a Quantum Gate into Circuit: add [GateType] [qidxes] [parameters]
            e.g. add h 2 \n
            Delete a Quantum Gate from Circuit: delete [index] \n
            Clean all gates in Circuit: clean \n
            Show Circuit Matrix: get_matrix \n
            Show Circuit's State Vector: get_state_vector \n
            Exit Circuit Debug: exit \n
        """
        print(help_str)

    def circuit_info(self):
        circuit_info = f"Circuit's Width: {self._circuit.width()} \n" + \
            f"Circuit's Size: {self._circuit.size()} \n" + \
            f"Circuit's Depth: {self._circuit.depth()} \n"
        gate_info = f"# of single-qubit gates: {self._circuit.count_1qubit_gate()} \n" + \
            f"# of bi-qubits gates: {self._circuit.count_2qubit_gate()} \n"

        print(circuit_info)
        print(gate_info)

    def initial_circuit(self, qubits: int):
        self._circuit = Circuit(qubits)

    def add_gate(self, gate_type: str, gate_info: list):
        gate_type = GateType[gate_type]
        gate = gate_builder(gate_type)
        gate_args = gate.controls + gate.targets

        qidxes = [int(gate_info[i]) for i in range(gate_args)]
        if gate_args < len(gate_info):
            params = [float(gate_info[j]) for j in range(gate_args, len(gate_info), 1)]
            gate.pargs = params

        gate | self._circuit(qidxes)

    def delete_gate(self, gate_index: int):
        self._circuit.pop(gate_index)

    def load_file(self, file_path):
        self._circuit = OPENQASMInterface.load_file(file_path).circuit

    def get_circuit_matrix(self):
        matrix = self._circuit.matrix()
        print(matrix)

    def get_state_vector(self):
        sv = self._simulation.run(self._circuit)
        print(sv)

    def run(self):
        print(CIRCUIT_DEBUG_DESCRIBE)
        start_show = False
        while True:
            input_str = input("please input circuit debug cmd:")
            variable = input_str.split(" ")
            print(variable)
            if variable[0] == "help":
                self.help_info()
                continue
            elif variable[0] == "load":
                self.load_file(variable[1])
                start_show = True
            elif variable[0] == "circuit":
                qubits = int(variable[1])
                self.initial_circuit(qubits)
                start_show = True
            elif variable[0] == "add":
                gate_type = variable[1]
                gate_info = variable[2:]
                self.add_gate(gate_type, gate_info)
            elif variable[0] == "delete":
                gate_index = int(variable[1])
                self.delete_gate(gate_index)
            elif variable[0] == "clean":
                qubits = self._circuit.width()
                self.initial_circuit(qubits)
                start_show = True
            elif variable[0] == "get_matrix":
                self.get_circuit_matrix()
                continue
            elif variable[0] == "get_state_vector":
                self.get_state_vector()
                continue
            elif variable[0] == "exit":
                break
            else:
                raise KeyError("Unsupportted Command.")

            if start_show:
                self._circuit.draw("command")
                self.circuit_info()

        print("End of the Circuit Debug.")


if __name__ == "__main__":
    entry_point = CircuitDebug()
    entry_point.run()
