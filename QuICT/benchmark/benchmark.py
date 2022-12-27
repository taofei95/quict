

from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator
from QuICT.tools.circuit_library.circuitlib import CircuitLib


class QuICTBenchmark:
    """ The QuICT Benchmarking. """
    def __init__(self, circuit_type: str, analysis_type: str, simulator=ConstantStateVectorSimulator()):
        """
        Initial circuit library

        Args:
            circuit_type (str, optional): one of [circuit, qasm, file]. Defaults to "circuit".
            output_result_type (str, optional): one of [Graph, table, txt]. Defaults to "Graph".
            simulator (Union[ConstantStateVectorSimulator, CircuitSimulator], optional): The simulator for simulating quantum circuit. Defaults to CircuitSimulator().
        """
        self._circuit_lib = CircuitLib(circuit_type)
        self._output_type = analysis_type
        self.simulator = simulator
        
    def _level_selection(self, level:str, qubit_num:int):
        circuits_list = []
        if level == "level1":
            fields_list = ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"]
            for field in fields_list:
                circuits = self._circuit_lib.get_benchmark_circuit(field, qubits_interval=qubit_num, max_size=qubit_num*10, max_depth=qubit_num*10)
                circuits_list.extend(circuits)

        elif level == "level2":
            fields_list = ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"]
            for field in fields_list:
                circuits = self._circuit_lib.get_benchmark_circuit(field, qubits_interval=qubit_num, max_size=qubit_num*10, max_depth=qubit_num*10)
                circuits_list.extend(circuits)
            circuits = self._circuit_lib.get_algorithm_circuit(["adder", "clifford", "qft"], qubits_interval=qubit_num, max_size=qubit_num*10, max_depth=qubit_num*10)
            circuits_list.extend(circuits)

        elif level == "level3":
            fields_list = ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"]
            for field in fields_list:
                circuits = self._circuit_lib.get_benchmark_circuit(field, qubits_interval=qubit_num, max_size=qubit_num*10, max_depth=qubit_num*10)
                circuits_list.extend(circuits)
            circuits = self._circuit_lib.get_algorithm_circuit(["adder", "clifford", "grover", "qft", "vqe", "cnf", "maxcut"], qubits_interval=qubit_num, max_size=qubit_num*10, max_depth=qubit_num*10)
            circuits_list.extend(circuits)

        return circuits_list
    
    def get_circuits(self, level, quantum_machine_info:int, mapping:bool, synthesis:bool):
        circuits = self._level_selection(level, quantum_machine_info)
        return circuits