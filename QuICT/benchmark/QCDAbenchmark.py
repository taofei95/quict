import random
from QuICT.benchmark.get_quantum_machine_circuit import QuantumMachineCircuitBuilder
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.utils.gate_type import GateType
from QuICT.tools.circuit_library import CircuitLib
from scipy.stats import unitary_group



class QCDAbenchmark:
    """ A benchmarking framework for quantum circuits automated design."""
    
    _qubit_list = [2, 4, 6, 8, 10, 15, 20]

    def _alg_circuit(self, bench_func):
        cirs_group = []
        alg_fields_list = ["adder", "clifford", "qft", "grover", "cnf", "maxcut", "qnn", "quantum_walk", "vqe"]
        for field in alg_fields_list:
            cirs = CircuitLib().get_circuit("algorithm", str(field), qubits_interval=self._qubit_list[-1])
            for cir in cirs:
                cir.gate_decomposition()
                cir.name = "+".join([bench_func, field, f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
                cirs_group.append(cir)
        return cirs_group

    def _machine_circuit(self, bench_func):
        cirs_group = []
        machine_fields_list = ["aspen-4", "ourense", "rochester", "sycamore", "tokyo"]
        for field in machine_fields_list:
            cirs = CircuitLib().get_circuit("machine", str(field), qubits_interval=self._qubit_list[-1])
            for cir in cirs:
                cir.name = "+".join([bench_func, field, f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
                cirs_group.append(cir)
        return cirs_group

    def _random_prob_circuit(self, bench_func):
        cirs_group = []
        prob_list = [0.2, 0.4, 0.6, 0.8, 1]
        one_qubit = [GateType.h, GateType.rx, GateType.ry, GateType.rz, GateType.x, GateType.y, GateType.z, GateType.u3]
        two_qubits = [GateType.cx, GateType.cz, GateType.swap]

        for prob in prob_list:
            for q in self._qubit_list:
                len_s, len_d = len(one_qubit), len(two_qubits)
                prob_list = [prob / len_s] * len_s + [(1 - prob) / len_d] * len_d
                cir = Circuit(q)
                cir.random_append(q * 10, typelist=one_qubit+two_qubits, probabilities=prob_list, seed=10)
                cir.name = "+".join([bench_func, "probrandom", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
                cirs_group.append(cir)
        return cirs_group

    def _clifford_pauli_circuit(self, bench_func):
        cirs_group = []
        for field in [CLIFFORD_GATE_SET, PAULI_GATE_SET]:
            for q in self._qubit_list:
                cir = Circuit(q)
                cir.random_append(q * 10, field)
                cir.name = "+".join([bench_func, "special", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
                cirs_group.append(cir)
        return cirs_group

    def _template_circuit(self, bench_func):
        cirs_group = []
        for q in self._qubit_list:
            cirs = CircuitLib().get_template_circuit(qubits_interval=q)
            for cir in cirs:
                sub_cir = cir.sub_circuit(qubit_limit=list(range(int(cir.width() / 2))))
                sub_cir | cir
                cir.name = "+".join([bench_func, "optimal", f"w{cir.width()}_s{cir.size()}_d{cir.depth()}"])
                cirs_group.append(cir)
        return cirs_group

    def _unitary_matrix(self, bench_func):
        bench_data = []
        for q in self._qubit_list:
            U = unitary_group.rvs(2 ** q)
            U.name = "+".join([bench_func, "umatrix", q])
            bench_data.append(U)
        return bench_data

    def _QSP(self, bench_func):
        bench_data = []
        for q in self._qubit_list:
            SV = np.random.random(1 << q).astype(np.complex128)
            SV.name = "+".join([bench_func, "statevector", q])

            real = np.random.random(q)
            imag = np.random.random(q)
            SV = (real + 1j * imag) / np.linalg.norm(real + 1j * imag)
            SV.name = "+".join([bench_func, "standard", q])

            SV = np.zeros(shape=(3, 3))
            SV.name = "+".join([bench_func, "zeros", q])

            SV = np.ones(shape=(3, 3))
            SV.name = "+".join([bench_func, "ones", q])

            SV = np.linspace(0.0, 10, num=q * q)
            SV.name = "+".join([bench_func, "degree_of_disparity", q])

            bench_data.append(SV)
        return bench_data

    def _mapping_bench(self, bench_func):
        circuits_list = []
        # algorithm circuit
        circuits_list.extend(self._alg_circuit(bench_func))

        # instruction set circuit
        circuits_list.extend(self._machine_circuit(bench_func))

        # quantum machine circuit
        qmc = QuantumMachineCircuitBuilder()
        cirs = qmc.get_machine_cir()
        circuits_list.extend(cirs)

        return circuits_list

    def _optimization_bench(self, bench_func):
        circuits_list = []

        # algorithm circuit
        # circuits_list.extend(self._alg_circuit(bench_func))

        # instruction set circuit
        # circuits_list.extend(self._machine_circuit(bench_func))

        # circuits with different probabilities of cnot
        circuits_list.extend(self._random_prob_circuit(bench_func))

        # # clifford / pauli instruction set circuit
        # circuits_list.extend(self._clifford_pauli_circuit(bench_func))

        # # Approaching the known optimal mapping circuit
        # circuits_list.extend(self._template_circuit(bench_func))

        return circuits_list

    def _gatetransform_bench(self, bench_func):
        # completely random circuits
        circuits_list = self._random_prob_circuit(bench_func)

        return circuits_list
    
    def _unitarydecomposition_bench(self, bench_func):
        # completely random circuits
        circuits_list = self._unitary_matrix(bench_func)

        return circuits_list

    def _quantumstatepreparation_bench(self, bench_func):
        # completely random circuits
        circuits_list = self._QSP(bench_func)

        return circuits_list

    def get_circuits(self, bench_func):
        """Get the circuit to be benchmarked

        Args:
            bench_func (str, optional): The type of qcdabenchmark.

        Returns:
            (List[Circuit]): Return the list of output circuit order by output_type.
        """

        if bench_func == "mapping":
            circuits_list = self._mapping_bench(bench_func)
        elif bench_func == "optimization":
            circuits_list = self._optimization_bench(bench_func)
        elif bench_func == "gatetransform":
            circuits_list = self._gatetransform_bench(bench_func)
        elif bench_func == "unitarydecomposition":
            circuits_list = self._unitarydecomposition_bench(bench_func)
        elif bench_func == "quantumstatepreparation":
            circuits_list = self._quantumstatepreparation_bench(bench_func)

        return circuits_list

