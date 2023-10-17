# pip install qiskit
# pip install qiskit-aer-gpu

from cmath import phase
import numpy as np
from qiskit import QuantumCircuit, transpile, Aer, QuantumRegister
from qiskit.quantum_info import Kraus, SuperOp
from qiskit.providers.aer import AerSimulator
from qiskit.tools.visualization import plot_histogram

# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError, ReadoutError
from qiskit.providers.aer.noise import pauli_error, amplitude_damping_error, phase_damping_error, phase_amplitude_damping_error
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.providers.aer.noise import thermal_relaxation_error

from qiskit.quantum_info import Operator


def test_pauil():
    p_error = 0.4
    # amp_error = amplitude_damping_error(0.1)
    # pauil error
    perror = pauli_error([('ZY', 0.4), ('XI', 0.6)])
    # print(Kraus(perror))
    # print("-----------------------------------------")
    # bit flip error
    bit_error = pauli_error([('X',p_error), ('I', 1 - p_error)])
    # print(error_gate1.to_dict())
    print("-----------------------------------------")
    # phase flip error
    phase_gate = pauli_error([('Z',p_error), ('I', 1 - p_error)])
    # print(Kraus(phase_gate))
    print("-----------------------------------------")

    com_gate = bit_error.tensor(bit_error)
    print(com_gate.to_dict())
    # print(Kraus(com_gate))
    # # phase-bit flip error
    # pb_gate = pauli_error([('Y',p_error), ('I', 1 - p_error)])
    # print(Kraus(pb_gate))
    # print("-----------------------------------------")
    
    noise_model = NoiseModel()
    # noise_model.add_all_qubit_quantum_error(bit_error, ['h'])
    # noise_model.add_all_qubit_quantum_error(phase_gate, ['h'])
    noise_model.add_quantum_error(perror, ['cx'], [0, 1])

    return noise_model


def test_depalor():
    error_1 = depolarizing_error(0.05, 1, False)
    # print(error_1.to_dict())
    print("-----------------------------------------")

    # com/ten
    com_err = error_1.tensor(error_1)
    # print(com_err.to_dict())

    error_2 = depolarizing_error(0.01, 2)
    # print(Kraus(error_2, True))
    
    noise_model = NoiseModel()
    noise_model.add_quantum_error(com_err, ['cx'], [0, 1])

    return noise_model


def test_damping():
    error_amp = amplitude_damping_error(0.2, 0.3, False)
    # print(error_amp.to_dict())
    print("-----------------------------------------")
    error_phase = phase_damping_error(0.5, False)
    # print(error_phase.to_dict())
    
    com_err = error_amp.tensor(error_phase)
    
    noise_model = NoiseModel()
    noise_model.add_quantum_error(com_err, ['cx'], [0, 1])

    return noise_model

    # print(com_err.to_dict())
    # print(com_err._op_shape)
    # com_err = error_amp.compose(error_phase)
    # print(com_err.to_dict())
    # print(com_err._op_shape)
    # print("-----------------------------------------")
    # error_ap = phase_amplitude_damping_error(0.1, 0.5, 0.2, False)
    # print(error_ap.to_dict())


def test_ops():
    p_error = 0.05
    error_gate1 = pauli_error([('X',p_error), ('I', 1 - p_error)])
    print(Kraus(error_gate1.dot(error_gate1)))
    print(Kraus(error_gate1.tensor(error_gate1)))


def build_qiskit_noise_model():
    # build noise model
    p_error = 0.4
    error_bf = pauli_error([('X',p_error), ('I', 1 - p_error)])
    # error_pf = pauli_error([('Y',p_error), ('I', 1 - p_error)])
    # bits_err = pauli_error([('XY', p_error), ('ZI', 1 - p_error)])
    # error_1 = depolarizing_error(1, 1)

    error_amp = amplitude_damping_error(0.1, 0.4, False)
    error_phase = phase_damping_error(0.3, False)
    error_ap = phase_amplitude_damping_error(0.1, 0.3, 0.5, False)
    noise_bit_flip = NoiseModel()
    noise_bit_flip.add_all_qubit_quantum_error(error_bf, ['h'])
    # noise_bit_flip.add_all_qubit_quantum_error(error_phase, ['u1', 'u3'])
    # noise_bit_flip.add_all_qubit_quantum_error(error_ap, ['x'])

    # noise_bit_flip.add_all_qubit_quantum_error(error_amp, ["h", "u1"])
    # noise_bit_flip.add_all_qubit_quantum_error(error_phase, ['y'])
    # noise_bit_flip.add_all_qubit_quantum_error(error_ap, ['x'])
    
    return noise_bit_flip


def build_qiskit_circuit():
    # Build Circuit
    circ = QuantumCircuit(10, 10)
    circ.h(9)
    circ.h(8)
    circ.h(7)
    circ.h(6)
    circ.h(5)
    circ.h(4)
    circ.h(3)
    circ.h(2)
    circ.h(1)
    circ.h(0)

    circ.x(8)
    circ.y(7)
    circ.z(6)
    circ.s(5)
    circ.sdg(4)
    circ.t(3)
    circ.tdg(2)
    circ.u1(np.pi / 2, 1)
    circ.u3(0, 0, np.pi / 2, 0)

    circ.cx(9,8)
    circ.cx(8,9)
    circ.cy(7,6)
    circ.cy(6,7)
    circ.cz(5,4)
    circ.cz(4,5)
    circ.ch(3,2)
    circ.ch(2,3)
    circ.crz(np.pi/2, 1,0)
    circ.crz(np.pi/2, 0,1)

    circ.h(0)
    circ.x(1)
    circ.y(2)
    circ.z(3)
    circ.s(4)
    circ.rx(np.pi / 2, 5)
    circ.t(6)
    circ.ry(np.pi / 2, 7)
    circ.u1(np.pi / 2, 8)
    circ.u2(np.pi / 2, np.pi / 2, 9)

    circ.rxx(np.pi,9,8)
    circ.rxx(np.pi,8,9)
    circ.ryy(np.pi,7,6)
    circ.ryy(np.pi,6,7)
    circ.rzz(np.pi,5,4)
    circ.rzz(np.pi,4,5)
    circ.cu1(np.pi / 2,3,2)
    circ.cu1(np.pi / 2,2,3)
    circ.cu3(np.pi / 2, 0, 1, 1, 0)
    circ.cu3(np.pi / 2, 1, 0, 0, 1)

    circ.swap(9,8)
    circ.swap(8,9)
    circ.rxx(np.pi,7,6)
    circ.rxx(np.pi,6,7)
    circ.ryy(np.pi,5,4)
    circ.ryy(np.pi,4,5)
    circ.rzz(np.pi,3,2)
    circ.rzz(np.pi,2,3)
    circ.swap(1,0)
    circ.swap(0,1)
    circ.ccx(9, 8, 7)
    circ.ccx(0, 1, 2)
    circ.cswap(6, 5, 4)
    circ.cswap(3, 4, 5)
    # circ.draw()
    # # circ.measure([0,1,2], [0,1,2])

    # circ.save_amplitudes(list(range(1 << 10)))
    # circ.save_density_matrix()

    simulator = Aer.get_backend('aer_simulator_statevector_gpu')
    circ = transpile(circ, simulator)

    # Run and get counts
    result = simulator.run(circ).result()


def build_ghz_circuit():
    circ = QuantumCircuit(4, 4)
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.cx(2, 3)
    circ.measure_all()

    return circ

noise_model = test_pauil()
circ = build_ghz_circuit()

sim_noise = AerSimulator(noise_model=noise_model, precision="double")
circ_tnoise = transpile(circ, sim_noise)

result = sim_noise.run(circ_tnoise, shots=1000).result()
data = result.get_counts(0)
print(data)

# Show the results
# data = result.data(0)["state_vector"]
print(result.data)
# np.save("density_matrix.npy", data)


def qiskit_unitary_decomposition():
    from qiskit import QuantumCircuit, transpile, Aer, QuantumRegister
    from scipy.stats import unitary_group
    
    matrix = unitary_group.rvs(2 ** 8)
    q_i = QuantumRegister(8)
    qcir = QuantumCircuit(q_i)
    qcir.isometry(
        isometry = matrix,
        q_input =q_i,
        q_ancillas_for_output = []
    )

    simulator = Aer.get_backend('aer_simulator')
    circ = transpile(qcir, simulator)

    # print(circ.qasm())
