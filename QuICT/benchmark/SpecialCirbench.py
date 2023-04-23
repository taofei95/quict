
import re
from QuICT.benchmark.benchmark import QuICTBenchmark
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.utils.gate_type import CLIFFORD_GATE_SET
from QuICT.simulation.simulator import Simulator
from QuICT.tools.circuit_library.circuitlib import CircuitLib
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *
from QuICT.simulation.state_vector import StateVectorSimulator
from QuICT.tools.circuit_library.get_mirror_circuit import MirrorCircuitBuilder


class SpecialCirbench:

    def __init__(self, width, size):
        self._width = width
        self._size = size
    
    def benchcir_evaluate(self):
        based_circuits_list = []
        based_fields_list = ["highly_entangled", "highly_parallelized", "highly_serialized", "mediate_measure"]
        for field in based_fields_list:
            circuits = CircuitLib().get_benchmark_circuit(str(field), qubits_interval=self._width)
            based_circuits_list.extend(circuits)

        eigenvalue_score = []
        for i in range(len(based_circuits_list)):
            field = based_circuits_list[i].split("+")[-2]
            cir_attribute = re.findall(r"\d+", based_circuits_list[i])
            if field == based_fields_list[0]:
                P = abs((int(cir_attribute[1]) / int(cir_attribute[2]) - 1) / (int(cir_attribute[0]) - 1))
                eigenvalue_score.append([based_circuits_list[i], P])
            elif field == based_fields_list[1]:
                M = (int(cir_attribute[3]) / int(cir_attribute[2]))
                eigenvalue_score.append([based_circuits_list[i], M])
            elif field == based_fields_list[2]:
                E = (1 - int(cir_attribute[3]) / int(cir_attribute[1]))
                eigenvalue_score.append([based_circuits_list[i], E])
            elif field == based_fields_list[3]:
                S = (1 - int(cir_attribute[3]) / int(cir_attribute[1]))
                eigenvalue_score.append([based_circuits_list[i], S])
        return eigenvalue_score

        
        
        
    def mirrorcir_evaluate(self,
            num_gates:int = 100,
            max_depth:int = 10,
            num_repeats:int = 10
            ):
        # Define circuit size depth and repeat counts
        # Define a set of gate list
        typelist = [ID, X, Y, Z, H, SX, CX]
        
        # Define an array to hold the fidelity results
        fidelity_results = np.zeros((num_gates, max_depth))

        for num_depth in range(1, max_depth + 1):
            for num_gate in range(num_gates):
               # Create a random gate sequence of the required depth
               sequence = []
               for _ in range(num_depth):
                   gate_idx = np.random.randint(0, len(typelist))
                   sequence.append(typelist[gate_idx])

            # build circuit
            cir = Circuit(2)

            # Apply the gate sequence to the circuit
            for gate in sequence:
                if gate == ID:
                    ID | cir(0)
                elif gate == X:
                    X | cir(0)
                elif gate == Y:
                    Y | cir(0)
                elif gate == Z:
                    Z | cir(0)
                elif gate == H:
                    H | cir(0)
                elif gate == SX:
                    SX | cir(0)
                elif gate == CX:
                    CX | cir([0, 1])
            
            # Add a mirror circuit to origin circuit
            mirror_circuit = MirrorCircuitBuilder().build_mirror_circuit(width=5, rand_unit=1, pro=0.8)
            mirror_circuit | cir
        print(cir.qasm())
        # simulate the circuit by quantum machine
        # result = QuICTBenchmark().bench_run()
        sim = Simulator()
        result = sim.run(cir, num_repeats)['data']['counts']

        # Compare the expected and actual outcomes to calculate the fidelity
        expected = '0' * num_repeats
        actual = ''.join([str(result[key]) for key in result])
        fidelity = (actual.count(expected) / num_repeats)

        # Add the fidelity value to the results array
        fidelity_results[num_gate, num_depth - 1] = fidelity
        
        # Calculate the average fidelity over all gate sequences for each sequence length
        average_fidelities = np.mean(fidelity_results, axis=0)
        
        return average_fidelities

    def qvcir_evaluate():
        pass
    def SRB_evaluate(self,
            num_gates:int = 100,
            max_depth:int = 10,
            num_repeats:int = 10
            ):
        # Define circuit size depth and repeat counts
        # Define a set of gate list
        typelist = [ID, X, Y, Z, H, SX, CX]
        
        # Define an array to hold the fidelity results
        fidelity_results = np.zeros((num_gates, max_depth))

        for num_depth in range(1, max_depth + 1):
            for num_gate in range(num_gates):
               # Create a random gate sequence of the required depth
               sequence = []
               for _ in range(num_depth):
                   gate_idx = np.random.randint(0, len(typelist))
                   sequence.append(typelist[gate_idx])

            # build circuit
            cir = Circuit(2)

            # Apply the gate sequence to the circuit
            for gate in sequence:
                if gate == ID:
                    ID | cir(0)
                elif gate == X:
                    X | cir(0)
                elif gate == Y:
                    Y | cir(0)
                elif gate == Z:
                    Z | cir(0)
                elif gate == H:
                    H | cir(0)
                elif gate == SX:
                    SX | cir(0)
                elif gate == CX:
                    CX | cir([0, 1])
            Measure | cir

        # simulate the circuit by quantum machine
        # result = QuICTBenchmark().bench_run()
        sim = Simulator()
        result = sim.run(cir, num_repeats)['data']['counts']

        # Compare the expected and actual outcomes to calculate the fidelity
        expected = '0' * num_repeats
        actual = ''.join([str(result[key]) for key in result])
        fidelity = (actual.count(expected) / num_repeats)

        # Add the fidelity value to the results array
        fidelity_results[num_gate, num_depth - 1] = fidelity
        
        # Calculate the average fidelity over all gate sequences for each sequence length
        average_fidelities = np.mean(fidelity_results, axis=0)
        
        return average_fidelities

if __name__ == "__main__":
    print(SpecialCirbench(5, 5).mirrorcir_evaluate())
        


# SpecialCirbench(2, 8).SRB_evaluate()



# build inverse clifford circuit
# cir = Circuit(self._width)
# rand_clifford = CompositeGate()
# rand_clifford_list = []
# for _ in range(int(self._size / 2)):
#     qubits_indexes = list(range(self._width))
#     gate_type = np.random.choice(CLIFFORD_GATE_SET)
#     gate = GATE_TYPE_TO_CLASS[gate_type]()
#     gate_size = gate.controls + gate.targets
#     gate & qubits_indexes[:gate_size] | rand_clifford
#     rand_clifford_list.append(rand_clifford)
# rand_clifford | cir
# for inv_gate in rand_clifford_list:
#     inv_clifford = inv_gate.inverse()
# inv_clifford | cir

# Measure | cir

# cir.draw(filename="aaa")

#########################################################################################
# from qiskit import *
# from qiskit.ignis.verification.randomized_benchmarking import randomized_benchmarking_seq

# # 定义量子线路
# qr = QuantumRegister(2)
# circ = QuantumCircuit(qr)

# # 添加Clifford门序列
# rb_opts = {}
# rb_opts[‘length_vector’] = [1, 10]
# rb_opts[‘seed’] = 10
# rb_opts[‘nseeds’] = 5
# rb_opts[‘rb_pattern’] = [[0, 1]]

# # 基于随机序列生成rb序列
# rb_circs, xdata = randomized_benchmarking_seq(**rb_opts)
# circ += rb_circs[0] # 选择一个随机序列

# # 模拟实际运行和模拟结果，并计算每个门序列的误差次数
# backend = Aer.get_backend(‘qasm_simulator’)
# job_sim = execute(circ, backend=backend, shots=1024)
# job_real = execute(circ, backend=IBMQ.get_backend(‘name_of_device’), shots=1024)
# job_monitor(job_real)

# # 计算每个门序列的误差次数
# error_counts = job_real.result().get_counts() - job_sim.result().get_counts()
# error_count = sum(error_counts.values())

# # 计算误差率
# total_count = sum(job_real.result().get_counts().values())
# error_rate = error_count / total_count

###############################################################################
# from qiskit import QuantumCircuit, execute, Aer
# from qiskit.providers.aer.noise import NoiseModel
# import numpy as np

# # Define a set of gates to test
# gate_set = [‘id’, ‘x’, ‘y’, ‘z’, ‘h’, ‘sx’, ‘cx’]

# # Create a noise model for the simulator
# # Noise can be added later and should be added based on the actual device noise characteristics
# noise_model = NoiseModel.from_backend(backend)

# # Define the maximum number of gate sequences
# num_sequences = 100

# # Define the maximum sequence length
# max_length = 10

# # Define the number of times to repeat each sequence
# num_repeats = 10

# # Define an array to hold the fidelity results
# fidelity_results = np.zeros((num_sequences, max_length))

# # Loop over the gate sequences
# for sequence_length in range(1, max_length + 1):
#     for sequence_num in range(num_sequences):
#         # Create a random gate sequence of the required length
#         sequence = []
#         for i in range(sequence_length):
#             gate_idx = np.random.randint(0, len(gate_set))
#             sequence.append(gate_set[gate_idx])

#         # Create the quantum circuit
#         qc = QuantumCircuit(1, 1)
#         qc.reset(0)

#         # Apply the gate sequence to the circuit
#         for gate_name in sequence:
#             if gate_name == ‘id’:
#                 qc.id(0)
#             elif gate_name == ‘x’:
#                 qc.x(0)
#             elif gate_name == ‘y’:
#                 qc.y(0)
#             elif gate_name == ‘z’:
#                 qc.z(0)
#             elif gate_name == ‘h’:
#                 qc.h(0)
#             elif gate_name == ‘sx’:
#                 qc.sx(0)
#             elif gate_name == ‘cx’:
#                 qc.cx(0, 1)

#         # Measure the qubit
#         qc.measure(0, 0)

#         # Execute the circuit on the simulator
#         job = execute(qc, Aer.get_backend(‘qasm_simulator’), noise_model=noise_model, shots=num_repeats)
#         result = job.result()

#         # Compare the expected and actual outcomes to calculate the fidelity
#         expected = ‘0’ * num_repeats
#         actual = ‘’.join([str(result.get_counts(qc)[key]) for key in result.get_counts(qc)])
#         fidelity = (actual.count(expected) / num_repeats)

#         # Add the fidelity value to the results array
#         fidelity_results[sequence_num, sequence_length - 1] = fidelity

# # Calculate the average fidelity over all gate sequences for each sequence length
# average_fidelities = np.mean(fidelity_results, axis=0)

# # Plot the results
# import matplotlib.pyplot as plt
# plt.plot(range(1, max_length + 1), average_fidelities)
# plt.xlabel(‘Sequence length’)
# plt.ylabel(‘Average fidelity’)
# plt.show()