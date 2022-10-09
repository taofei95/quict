import time
from QuICT.algorithm.quantum_algorithm import ShorFactor
from QuICT.simulation.state_vector import ConstantStateVectorSimulator

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.algorithms import Shor
from qiskit import QuantumCircuit

f = open("qiskit_alg_benchmark_shor_data.txt", "w+")
run_test_modes = ["BEA_zip"]
number_list = [
    15,
    21,
    # 33,
    # 35,
]
# simulator = ConstantStateVectorSimulator()

backend = Aer.get_backend('aer_simulator')
quantum_instance = QuantumInstance(backend, shots=1)

for mode in run_test_modes:
    f.write(f"mode: {mode}\n")
    for number in number_list:
        f.write(f"qubits: {number}\n")
        stime = time.time()
        a = ShorFactor(mode=mode).run(N=number, forced_quantum_approach=True)
        ttime = time.time()
        cir = ShorFactor(mode=mode).circuit(N=number)[0]
        f.write(f"Quict time : {round(ttime - stime, 6)}\n")
        f.write(f"Quict qubits : {cir.width()}\n")

        sstime = time.time()
        shor = Shor(quantum_instance=quantum_instance)
        result = shor.factor(number)
        tttime = time.time()
        f.write(f"Qiskit time : {round(tttime - sstime, 6)}\n")
        f.write(f"Qiskit qubits : {shor.construct_circuit(number).num_qubits}\n")

        # f.write(f"Quict time : {ttime - stime}, Qiskit time : {tttime - sstime}\n")
        # f.write(f"Quict qubits : {cir.width()}, Qiskit qubits : {shor.construct_circuit(number).num_qubits}\n")

f.close()