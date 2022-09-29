import os
import time

from QuICT.simulation.simulator import Simulator
from QuICT.simulation.state_vector import CircuitSimulator, ConstantStateVectorSimulator
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

from qiskit import QuantumCircuit, Aer
from qiskit.execute_function import execute
from qiskit_aer import *

qubits_num = [20]
gates_num = [5, 7, 9, 11, 13, 15]
sim_c = CircuitSimulator()
sim_g = ConstantStateVectorSimulator(gpu_device_id=0)
backend_c = Aer.get_backend('statevector_simulator')
# backend_g = StatevectorSimulator()
# backend_g = AerSimulator(methods='statevector', device='GPU')
backend_g = Aer.get_backend('aer_simulator')
# backend_g = Aer.get_backend('aer_simulator_statevector_gpu')

f = open("qiskit_state_vector_speed.txt", 'w+')
for q_num in qubits_num:
    f.write(f"qubit_number: {q_num} \n")
    circuit_folder_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "circuit/qiskitnew"
    )
    for gm in gates_num:
        # f.write(f"gate size: {q_num * gm} \n")
        quict_cpu_time, quict_gpu_time, qiskit_cpu_time, qiskit_gpu_time = 0, 0, 0, 0
        for i in range(10):
            filename = f"q{q_num}-g{gm * q_num}-{i}.qasm"
            cir = OPENQASMInterface.load_file(
                circuit_folder_path + '/' + filename
            ).circuit

            # quict cpu
            # stime = time.time()
            # sv = sim_c.run(cir)
            # ltime = time.time()
            # quict_cpu_time += round(ltime - stime, 6)

            # # quict gpu
            # sstime = time.time()
            # sv = sim_g.run(cir)
            # lltime = time.time()
            # quict_gpu_time += round(lltime - sstime, 6)

            # # qiskit cpu
            circ = QuantumCircuit.from_qasm_file(circuit_folder_path + '/' + filename)
            # ssstime = time.time()
            # job = backend_c.run(circ)
            # llltime = time.time()
            # qiskit_cpu_time += round(llltime - ssstime, 6)

            # qiskit gpu
            # backend_g = AerSimulator()
            # backend_g['method'] = 'statevector'
            # backend_g['device'] = 'GPU'
            # simulator.backend_options(methods='statevector', device='GPU')
            # result = execute(circ, backend_g, backend=AerSimulator).result()
            # print(result)

            backend_g.set_options(device='GPU')
            sssstime = time.time()
            job = backend_g.run(circ)
            lllltime = time.time()
            qiskit_gpu_time += round(lllltime - sssstime, 6)

        f.write(f"quict cpu time : {round(quict_cpu_time/10, 6)}, quict gpu time : {round(quict_gpu_time/10, 6)}, qiskit cpu time : {round(qiskit_cpu_time/10, 6)}, qiskit gpu time : {round(qiskit_gpu_time/10, 6)}\n")

f.close()
