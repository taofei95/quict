import os
import random
from QuICT.benchmark import QuICTBenchmark
from QuICT.core import Circuit
from QuICT.core.gate import *
from scipy.stats import unitary_group
from QuICT_ml.rl_mapping import RlMapping
from QuICT.qcda.mapping.mcts import MCTSMapping
from QuICT.qcda.mapping.sabre import SABREMapping
from QuICT.core.utils.gate_type import GateType
from QuICT.core.layout.layout import Layout
from QuICT.qcda.optimization import clifford_rz_optimization, cnot_without_ancilla, commutative_optimization, symbolic_clifford_optimization, template_optimization
from QuICT.qcda.optimization.commutative_optimization.commutative_optimization import CommutativeOptimization
from QuICT.qcda.synthesis import gate_transform
from QuICT.qcda.synthesis.gate_transform.gate_transform import GateTransform
from QuICT.qcda.synthesis.quantum_state_preparation.quantum_state_preparation import QuantumStatePreparation
from QuICT.qcda.synthesis.unitary_decomposition.unitary_decomposition import UnitaryDecomposition
from QuICT.tools.circuit_library.circuitlib import CircuitLib
from unit_test.qcda.synthesis.quantum_state_preparation.quantum_state_preparation_unit_test import random_unit_vector


class QCDAbench:

    def __init__(self, width, size, level):
        self._width = width
        self._size = size
        if level is not None:
            _circuits = self._circuit_selection(level)

    _based_fields_list = ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure", "mirror", "qv"]
    _alg_fields_list = ["adder", "clifford", "qft", "grover", "cnf", "maxcut", "qnn", "quantum_walk", "vqe", "qpe"]
    _ran_fields_list = [
        "aspen-4", "ourense", "rochester", "sycamore", "tokyo", "ctrl_unitary", "diag",
        "single_bit", "ctrl_diag", "google", "ibmq", "ionq", "ustc", "nam", "origin"
        ]

    def _circuit_selection(self, level):
        based_circuits_list = []
        for field in self._based_fields_list:
            circuits = CircuitLib().get_benchmark_circuit(str(field), qubits_interval=self._width )
            based_circuits_list.extend(circuits)
        random_fields_list = random.sample(self._ran_fields_list, 5)
        for field in random_fields_list:
            circuits = CircuitLib().get_random_circuit(str(field), qubits_interval=self._width )
            based_circuits_list.extend(circuits)
        if level >= 2:
            for field in self._alg_fields_list[:3]:
                circuits = CircuitLib().get_algorithm_circuit(str(field), qubits_interval=self._width )
                based_circuits_list.extend(circuits)
        if level >= 3:
            for field in self._alg_fields_list[3:]:
                circuits = CircuitLib().get_algorithm_circuit(str(field), qubits_interval=self._width )
                based_circuits_list.extend(circuits)

        return based_circuits_list

    def _circuit_construct(self, cir_type="random"):
        """ Construct random circuit. """
        # Build Circuit
        circuit = Circuit(self._width)
        if cir_type == "random":
            circuit.random_append(self._size)
        if cir_type == "clifford":
            circuit.random_append(self._size, CLIFFORD_GATE_SET)
        if cir_type == "cnot":
            circuit.random_append(self._size, [GateType.cx])

        return circuit

    def mappingbench(self, layout=None, map_func=None):
        _map_func = [RlMapping, SABREMapping, MCTSMapping]
        assert map_func in _map_func, "please check mapping function"
        # circuit model
        circuit = self._circuit_construct()
        # layout
        if layout is None:
            layout = Layout.load_file(os.path.dirname(os.path.abspath(__file__)) + f"/example/layout/grid_3x3.json")
        else:
            layout = Layout.load_file(layout)
        # mapping function
        if map_func is None:
            map_func = random.choice(_map_func)
        mapper = map_func(layout)
        # circuit after mapping
        circuit_map = mapper.execute(circuit)
        
        bench_result = circuit_map.count_gate_by_gatetype([GateType.swap])

        return circuit, circuit_map, bench_result
    
    def optimizebench(self, optimizer_func=None):
        # circuit model
        _optimizer_func = [
            CommutativeOptimization,
            symbolic_clifford_optimization,
            clifford_rz_optimization,
            cnot_without_ancilla,
            template_optimization
        ]
        assert optimizer_func in _optimizer_func, "please check optimization function"
        if optimizer_func == _optimizer_func[0] or optimizer_func == _optimizer_func[1] or optimizer_func == _optimizer_func[4]:
            circuit = self._circuit_construct()
            circuit_opt = _optimizer_func().execute(circuit)
        if optimizer_func == _optimizer_func[2]:
            circuit = self._circuit_construct("clifford")
            circuit_opt = _optimizer_func().execute(circuit)
        if optimizer_func == _optimizer_func[3]:
            circuit = self._circuit_construct("cnot")
            circuit_opt = _optimizer_func().execute(circuit)

        bench_result = [
            circuit_opt.width(),
            circuit_opt.size(),
            circuit_opt.depth(),
            circuit_opt.count_1qubit_gate(),
            circuit_opt.count_2qubit_gate()
        ]

        return circuit, circuit_opt, bench_result

    def synthesisbench(self, synthesis_func):
        _synthesis_func = [UnitaryDecomposition, QuantumStatePreparation, gate_transform]
        _InSet = ["GoogleSet", "IBMQSet", "IonQSet", "NamSet", "OriginSet", "USTCSet"]

        assert synthesis_func in _synthesis_func, "please check synthesis function"

        if synthesis_func == _synthesis_func[0]:
            matrix = unitary_group.rvs(2 ** self._width)
            circuit_syn, _ = synthesis_func.execute(matrix)
        if synthesis_func == _synthesis_func[1]:
            state_vector = random_unit_vector(1 << self._width)
            gates = synthesis_func.execute(state_vector)
            circuit_syn = Circuit(self._width)
            circuit_syn.extend(gates)
        else:
            circuit = self._circuit_construct()
            for InSet in _InSet:
                GT = GateTransform(InSet)
                circuit_syn = GT.execute(circuit)

        bench_result = [circuit_syn.size(), circuit_syn.depth()]

        return circuit_syn, bench_result

    def score(circuits_list):
        QuICTBenchmark().bench_run(circuits_list)