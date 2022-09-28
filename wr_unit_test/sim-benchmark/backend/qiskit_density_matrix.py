import os
import time
from QuICT.simulation.density_matrix.density_matrix_simulator import DensityMatrixSimulation

from QuICT.simulation.state_vector import CircuitSimulator
from QuICT.simulation.density_matrix import DensityMatrixSimulation
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

from qiskit import QuantumCircuit, Aer, transpile

qubits_num = [4, 6, 8, 10, 12]
gates_num = [5, 7, 9, 11, 13, 15]
sim_c = CircuitSimulator()
sim_g = DensityMatrixSimulation("GPU")
sim_q_c = Aer.get_backend('aer_simulator_density_matrix')
simu_q_g = Aer.get_backend('aer_simulator_density_matrix')

f = open("qiskit_density_matrix_speed.txt", 'w+')
for q_num in qubits_num:
    f.write(f"qubit_number: {q_num} \n")
    circuit_folder_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "circuit/qiskitdm"
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
            stime = time.time()
            sv = sim_c.run(cir)
            ltime = time.time()
            quict_cpu_time += round(ltime - stime, 6)

            # quict gpu
            sstime = time.time()
            sv = sim_g.run(cir)
            lltime = time.time()
            quict_gpu_time += round(lltime - sstime, 6)
            
            # qiskit cpu
            circ = QuantumCircuit.from_qasm_file(circuit_folder_path + '/' + filename)
            circ = transpile(circ, sim_q_c)
            ssstime = time.time()
            amp = sim_q_c.run(circ)
            llltime = time.time()
            qiskit_cpu_time += round(llltime - ssstime, 6)

            #qiskit gpu
            simu_q_g.set_options(device='GPU')
            circ = transpile(circ, simu_q_g)
            sssstime = time.time()
            amp = simu_q_g.run(circ)
            lllltime = time.time()
            qiskit_gpu_time += round(lllltime - sssstime, 6)

        f.write(f"quict cpu time : {round(quict_cpu_time/10, 6)}, quict gpu time : {round(quict_gpu_time/10, 6)}, qiskit cpu time : {round(qiskit_cpu_time/10, 6)}, qiskit gpu time : {round(qiskit_gpu_time/10, 6)}\n")
        
f.close()
