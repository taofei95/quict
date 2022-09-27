import os
import time

from QuICT.simulation.simulator import Simulator
from QuICT.simulation.state_vector import CircuitSimulator, ConstantStateVectorSimulator
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

from qiskit import QuantumCircuit, Aer

qubits_num = [5, 10, 15, 20]
gates_num = [5, 7, 9, 11, 13, 15]
sim_c = CircuitSimulator()
sim_g = ConstantStateVectorSimulator(gpu_device_id=0)
# backend = Aer.get_backend('statevector_simulator')

f = open("qiskit_state_vector_speed.txt", 'w+')
for q_num in qubits_num:
    f.write(f"qubit_number: {q_num} \n")
    circuit_folder_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "circuit/qiskitnew"
    )
    for gm in gates_num:
        # f.write(f"gate size: {q_num * gm} \n")
        quict_cpu_time, quict_gpu_time = 0, 0
        for i in range(10):
            filename = f"q{q_num}-g{gm * q_num}-{i}.qasm"
            cir = OPENQASMInterface.load_file(
                circuit_folder_path + '/' + filename
            ).circuit

            # quict cpu
            stime = time.time()
            sv = sim_c.run(cir)
            ltime = time.time()
            quict_cpu_time += round(ltime - stime, 6)
            # f.write(f"{i},Quict run time cpu : {round(ltime - stime, 6)}\n")

            # quict gpu
            sstime = time.time()
            sv = sim_g.run(cir)
            lltime = time.time()
            quict_gpu_time += round(lltime - sstime, 6)
            # f.write(f"{i},Quict run time gpu : {round(lltime - sstime, 6)}\n")

            #qiskit
            # circ = QuantumCircuit.from_qasm_file(circuit_folder_path + '/' + filename)
            # # circ.save_amplitudes()
            # # circ.save_statevector()
            # ssstime = time.time()
            # job = backend.run(circ)
            # # result = job.result()
            # # outputstate = result.get_statevector(circ)
            # llltime = time.time()
            # qiskit_cpu_time += round(llltime - ssstime, 6)
            # f.write(f"{i},Qiskit run time : {round(llltime - ssstime, 6)}\n")
        # f.write(f"quict average simulation time : {quict_cpu_time/10}, qiskit average simulation time : {qiskit_cpu_time/10} \n")
        f.write(f"quict cpu average simulation time : {round(quict_cpu_time/10, 6)}, quict gpu average simulation time : {round(quict_gpu_time/10, 6)}\n")

f.close()
