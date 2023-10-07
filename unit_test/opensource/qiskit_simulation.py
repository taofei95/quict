# pip install qiskit
# pip install qiskit-aer-gpu

import os, csv
import time

from qiskit import Aer, qasm2, transpile
from qiskit.qasm2 import LEGACY_CUSTOM_INSTRUCTIONS


def _get_all_files(folder: str):
    file_path = os.path.join(os.path.dirname(__file__), f"../platform_compare/{folder}")
    files = os.listdir(file_path)
    files.sort()

    return file_path, files


def _files_name_extract(file_name: str, max_qubits):
    width, size, _, _ = file_name.split('_')

    if max_qubits is not None:
        int_width = int(width[1:])
        if int_width > max_qubits:
            return False

    return width + size


def load_circuit_from_qasm(folder: str, device: str, max_qubits = None):
    # load qasm
    file_path, files = _get_all_files(folder)
    simulator = Aer.get_backend('aer_simulator_statevector')
    if device == "GPU":
        simulator.set_options(device='GPU')

    time_combined = {}
    for f in files:
        cir_info = _files_name_extract(f, max_qubits)
        if not cir_info:
            continue

        # generate circuit
        fpath = os.path.join(file_path, f)
        circuit = qasm2.load(fpath, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
        circuit.save_statevector()

        stime = time.time()
        circ = transpile(circuit, simulator)
        sv = simulator.run(circ).result()
        # print(sv.data(0)['counts'])
        # sv.get_statevector(circ)
        etime = time.time()

        if cir_info not in time_combined.keys():
            time_combined[cir_info] = 0

        time_combined[cir_info] += (etime - stime) / 10

    return time_combined


def load_alg_circuit(folder: str, device: str, repeat: int = 10):
    # load qasm
    file_path, files = _get_all_files(folder)
    simulator = Aer.get_backend('aer_simulator_statevector')
    if device == "GPU":
        simulator.set_options(device='GPU')

    time_combined = {}
    for f in files:
        if device == "GPU" and f.startswith("bb84"):
            continue

        if device == "CPU" and f.startswith("wstate"):
            continue

        # generate circuit
        fpath = os.path.join(file_path, f)
        circuit = qasm2.load(fpath, custom_instructions=LEGACY_CUSTOM_INSTRUCTIONS)
        # circuit.save_statevector()
        # circ = transpile(circuit, simulator)

        total_time = 0
        for _ in range(repeat):
            stime = time.time()
            _ = simulator.run(circuit).result()
            etime = time.time()
            total_time += (etime - stime) / repeat

        time_combined[f[:-5]] = total_time

    return time_combined


if __name__ == "__main__":
    curr_path = os.getcwd()
    output_path = os.path.join(curr_path, "platform_result", "qiskit")
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Machine Result
    machine_list = ["origin6"] #, "ibm_manila", "ibm_quito", "ibm_perth", "ibmq_guadalupe", "quafu10", "quafu18"]
    device_list = ["CPU", "GPU"]

    for device in device_list:
        file = open(f"{output_path}/machine_result_{device}.csv", "w", newline="")
        writer = csv.writer(file)
        writer.writerow(["Machine Name", 10, 20, 50, 100])
        for machine in machine_list:
            row = [machine]
            tc = load_circuit_from_qasm(machine, device)
            time_spend = list(tc.values())
            time_spend.sort()
            writer.writerow(row + time_spend)

    # # Random Result
    # min_qubits = 5
    # for device in device_list:
    #     file = open(f"{output_path}/random_result_{device}.csv", "w", newline="")
    #     writer = csv.writer(file)
    #     writer.writerow(["Qubit Number", 10, 20, 50, 100])
    #     max_qubits = 25 if device == "CPU" else 30
    #     tc = load_circuit_from_qasm('random_circuit', device, max_qubits)

    #     for q in range(min_qubits, max_qubits + 1):
    #         row = [q]
    #         for g in [10, 20, 50, 100]:
    #             row.append(tc[f"w{q}s{g * q}"])

    #         writer.writerow(row)

    # # Algorithm Result
    # alg_folder = ["qasmbench/small", "qasmbench/medium"]
    # for device in device_list:
    #     file = open(f"{output_path}/algorithm_result_{device}.csv", "w", newline="")
    #     writer = csv.writer(file)
    #     writer.writerow(["Algorithm Name", "Running Time"])

    #     for folder in alg_folder:
    #         alg_result = load_alg_circuit(folder, device, repeat = 10)

    #         for key, val in alg_result.items():
    #             writer.writerow([key, val])


from QuICT.core.gate import *
from QuICT.core.utils.gate_type import GateType

# gate with one qubit
one_qubits_gate = [
    GateType.h, GateType.s, GateType.sdg, GateType.x, GateType.y, GateType.z,
    GateType.sx, GateType.sy, GateType.sw, GateType.id, GateType.u1, GateType.u2, GateType.u3,
    GateType.rx, GateType.ry, GateType.rz, GateType.t, GateType.tdg, GateType.phase,
    GateType.measure, GateType.reset, GateType.barrier
]
# gate with two qubits
two_qubits_gate = [
    GateType.cx, GateType.cz, GateType.ch, GateType.crz, GateType.cu1, GateType.cu3, GateType.fsim,
    GateType.rxx, GateType.ryy, GateType.rzz, GateType.swap, GateType.iswap
]
# gate with three qubits
three_qubits_gate = [GateType.ccx, GateType.ccz, GateType.cswap]
# gate with params
parameter_gates_for_call_test = [U1, U2, CU3, FSim]