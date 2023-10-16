import random
import numpy as np

from QuICT.core.circuit import Circuit
from QuICT.core.gate import *

from qiskit import transpile, qasm2, Aer
from QuICT.simulation.state_vector.statevector_simulator import StateVectorSimulator


common_gates = [
    GateType.h, GateType.s, GateType.sdg, GateType.x, GateType.y, GateType.z,
    GateType.id, GateType.u1, GateType.u2, GateType.u3, GateType.rx, GateType.ry,
    GateType.rz, GateType.t, GateType.tdg, GateType.cx, GateType.cz, GateType.cu1,
    GateType.cu3, GateType.rxx, GateType.swap, GateType.ccx, GateType.cswap
]

# test simulation accurancy with qiskit
def common_gates_random_circuit(q):
    cir_quict = Circuit(q)
    cir_qiskit = Circuit(q)
    qubit_range = list(range(q))
    for gate in common_gates:
        gate = gate_builder(gate, random_params=True)
        gate_index = gate.controls + gate.targets
        index_quict = list(random.sample(qubit_range, gate_index))
        index_qiskit = []
        for i in range(1, gate_index + 1):
            index_qiskit.append(q - 1 - index_quict[i - 1])

        gate & index_quict | cir_quict
        gate & index_qiskit | cir_qiskit
    f1 = open("unit_test/opensource/data/random_circuit_for_correction.qasm", "w+")
    f1.write(cir_quict.qasm())
    f1.close()
    f2 = open("unit_test/opensource/data/qiskit.qasm", "w+")
    f2.write(cir_qiskit.qasm())
    f2.close()
    return cir_quict, cir_qiskit

def special_random_circuit(q):
    cir_quict = Circuit(q)
    cir_qiskit = Circuit(q)
    cgate1, cgate2 = common_gates_random_circuit(q)
    cgate1.to_compositegate()
    cgate2.to_compositegate()

    for _ in range(3):
        cgate1 | cgate1
    cgate1 | cir_quict
    for _ in range(3):
        cgate2 | cgate2
    cgate2 | cir_qiskit
    f1 = open("unit_test/opensource/data/quict.qasm", "w+")
    f1.write(cir_quict.qasm())
    f1.close()
    f2 = open("unit_test/opensource/data/qiskit.qasm", "w+")
    f2.write(cir_qiskit.qasm())
    f2.close()
    return cir_quict, cir_qiskit

def test_simulation_accurancy():
    cir_quict, _ = common_gates_random_circuit(5)
    # cir_quict, _ = special_random_circuit(5)
    cir_quict.inverse() | cir_quict
    sim = StateVectorSimulator()
    SV = sim.run(cir_quict)
    print(SV)

def test_simulation_with_qiskit():
    common_gates_random_circuit(5)

    # qiskit:read qasm 
    cir_qiskit = qasm2.load(
        filename="unit_test/opensource/data/qiskit.qasm",
        include_path=qasm2.LEGACY_INCLUDE_PATH,
        custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS,
        custom_classical=qasm2.LEGACY_CUSTOM_CLASSICAL,
    )
    # func3:state_vector simulation and save
    backend1 = Aer.get_backend('statevector_simulator')
    job1 = backend1.run(cir_qiskit)
    result1 = job1.result()
    sv_data = result1.get_statevector(cir_qiskit)
    save_path = 'unit_test/opensource/data/state_vector.npy'
    np.save(save_path, sv_data)

    # func4:density_matrix simulation and save
    backend2 = Aer.get_backend('aer_simulator_density_matrix')
    cir_qiskit = transpile(cir_qiskit, backend2)
    cir_qiskit.save_density_matrix()
    job2 = backend2.run(cir_qiskit)
    result2 = job2.result()
    dm_data = result2.data()["density_matrix"]
    save_path = 'unit_test/opensource/data/density_matrix.npy'
    np.save(save_path, dm_data)

test_simulation_with_qiskit()

