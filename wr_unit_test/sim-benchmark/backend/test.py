import os
import time

from QuICT.simulation.state_vector.gpu_simulator.constant_statevector_simulator import ConstantStateVectorSimulator
from QuICT.tools.interface.qasm_interface import OPENQASMInterface

from qiskit import QuantumCircuit, Aer

qubits_num = [5]
gates_num = [5, 7]

f = open("speed.txt", 'w+')
for q_num in qubits_num:
    f.write(f"qubit_number: {q_num} \n")
    circuit_folder_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "circuit/qiskit"
    )
    for gm in gates_num:
        f.write(f"gate size: {q_num * gm} \n")
        for i in range(10):
            filename = f"q{q_num}-g{gm * q_num}-{i}.qasm"
            cir = OPENQASMInterface.load_file(
                circuit_folder_path + '/' + filename
            ).circuit
            # #quict
            # stime = time.time()
            # sim = ConstantStateVectorSimulator()
            # sv = sim.run(cir).get()
            # ltime = time.time()
            #qiskit
            circ = QuantumCircuit.from_qasm_file(circuit_folder_path + '/' + filename)
            # circ_opt = QasmBackendConfiguration(circuits=circ, backend_name=)
            stime = time.time()
            backend = Aer.get_backend('statevector_simulator')
            job = backend.run(circ)
            print(job)
            result = job.result()
            outputstate = result.get_statevector(circ)
            ltime = time.time()
            f.write(f"{i},QuICT run time : {round(ltime - stime, 6)}\n")
