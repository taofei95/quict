from qiskit import transpile, QuantumCircuit
from qiskit.providers.aer import AerSimulator

from QuICT.core.noise.noise_error import BitflipError, DampingError, PauliError
from QuICT.core.noise.noise_model import NoiseModel
from QuICT.simulation.density_matrix.density_matrix_simulator import DensityMatrixSimulation
from QuICT.tools.interface.qasm_interface import OPENQASMInterface
# from wr_unit_test.core.qiskit_noise_test import build_qiskit_noise_model

###############################################################
cir = OPENQASMInterface.load_file("wr_unit_test/qiskit_sim_test/random_circuit_for_correction.qasm").circuit
pauil_error_rate = 0.4
bf_err = BitflipError(pauil_error_rate)

nm = NoiseModel()
amp_err = DampingError(amplitude_prob=0.1, phase_prob=0, dissipation_state=0.4)
nm.add_noise_for_all_qubits(bf_err, ['s'])

sim = DensityMatrixSimulation(accumulated_mode=True)
result = sim.run(cir, noise_model=nm)
print(result)


#################################################################
# circ = QuantumCircuit.from_qasm_file("wr_unit_test/qiskit_sim_test/qiskit_random_circuit_for_correction.qasm")
# noise_bit_flip = build_qiskit_noise_model()
# sim_noise = AerSimulator(noise_model=noise_bit_flip, precision="double")
# circ_tnoise = transpile(circ, sim_noise)
# result = sim_noise.run(circ_tnoise, shots=1).result()

# print(result.get_counts(0))
