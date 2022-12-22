
from QuICT.core.noise.noise_error import BitflipError, DampingError, DepolarizingError, PauliError, PhaseflipError
from qiskit.providers.aer.noise import depolarizing_error
from qiskit.quantum_info import Kraus, SuperOp
from QuICT.core.noise.noise_model import NoiseModel
from QuICT.simulation.density_matrix.density_matrix_simulator import DensityMatrixSimulation

from QuICT.tools.interface.qasm_interface import OPENQASMInterface

def test_depalor():
    error_1 = depolarizing_error(0.05, 1)
    cir = OPENQASMInterface.load_file("unit_test/simulation/data/random_circuit_for_correction.qasm").circuit

    depolarizing_rate = 0.05
    single_dep = DepolarizingError(depolarizing_rate, num_qubits=1)
    
    nm = NoiseModel()
    nm.add_noise_for_all_qubits(single_dep, ['h', 'u1'])
    dm_simu = DensityMatrixSimulation()
    quict_result = dm_simu.run(cir, noise_model=nm)
    
    print(quict_result)
    print(Kraus(error_1, True))
    print("-----------------------------------------")
    
test_depalor()