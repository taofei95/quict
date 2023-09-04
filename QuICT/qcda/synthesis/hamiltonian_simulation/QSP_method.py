from QuICT.core.gate import *
from QuICT.qcda.synthesis import UnitaryDecomposition
from QuICT.core import Circuit
import numpy as np

def block_encoding(Hamiltonian):
    eigenvalue, eigenvector = np.linalg.eig(Hamiltonian)
    conj_matrix = []
    for i in range(len(eigenvalue)):
        transpose_eigenvector = eigenvector[i].reshape(len(eigenvector[i]),1)
        conj_matrix.append(np.sqrt(1-eigenvalue[i]**2)*np.kron(transpose_eigenvector, eigenvector[i]))
    conj_matrix = np.sum(np.array(conj_matrix), axis = 0)

    temp_matrix = np.hstack((conj_matrix, -1 * Hamiltonian))
    block_encoding_matrix = np.hstack((Hamiltonian, conj_matrix))
    block_encoding_matrix = np.vstack((block_encoding_matrix, temp_matrix))
    UD = UnitaryDecomposition()
    encoding_gates, _ = UD.execute(block_encoding_matrix)
    circuit_size = int(np.log2(len(block_encoding_matrix[0])))
    return encoding_gates, circuit_size

def projector_controller(ancilla_size, angle):
    mct = MultiControlToffoli("no_aux")
    mct_gates =  mct(ancilla_size)
    cg = CompositeGate()
    cg_x = CompositeGate()
    for i in range(ancilla_size):
        X | cg_x(i)

    #compose projector controller phase shift operator in equation (26)
    cg_x | cg([i for i in range(ancilla_size)])
    mct_gates | cg([i for i in range(ancilla_size+1)])
    Rz(2*angle) | cg(ancilla_size)
    mct_gates | cg([i for i in range(ancilla_size+1)])
    cg_x | cg([i for i in range(ancilla_size)])
    return cg
def build_circuits(Hamiltonian, angle_sequence):
    Hamiltonian_encoding, num_ancilla_qubit = block_encoding(Hamiltonian)
    Hamiltonian_encoding_inverse = Hamiltonian_encoding.inverse()
    circuit = Circuit(num_ancilla_qubit+1)
    if len(angle_sequence)%2 == 0: # if even parity
        for i in range(int(len(angle_sequence)/2)):
            print(angle_sequence[-2*i-1], angle_sequence[-2*i-2])
            Hamiltonian_encoding | circuit([j for j in range(num_ancilla_qubit)])
            projector_controller_gates = projector_controller(num_ancilla_qubit, angle_sequence[-2*i-1])
            projector_controller_gates | circuit([j for j in range(num_ancilla_qubit+1)])
            Hamiltonian_encoding_inverse | circuit([j for j in range(num_ancilla_qubit)])
            projector_controller_gates  = projector_controller(num_ancilla_qubit, angle_sequence[-2*i-2])
            projector_controller_gates | circuit([j for j in range(num_ancilla_qubit+1)])
    if len(angle_sequence)%2 == 1:
        for i in range(int((len(angle_sequence)-1) / 2)):
            print(angle_sequence[-2 * i - 1], angle_sequence[-2 * i - 2])
            Hamiltonian_encoding | circuit([j for j in range(num_ancilla_qubit)])
            projector_controller_gates = projector_controller(num_ancilla_qubit, angle_sequence[-2*i-1])
            projector_controller_gates | circuit([j for j in range(num_ancilla_qubit+1)])
            Hamiltonian_encoding_inverse | circuit([j for j in range(num_ancilla_qubit)])
            projector_controller_gates  = projector_controller(num_ancilla_qubit, angle_sequence[-2*i-2])
            projector_controller_gates | circuit([j for j in range(num_ancilla_qubit+1)])

        Hamiltonian_encoding | circuit([j for j in range(num_ancilla_qubit)])
        projector_controller_gates = projector_controller(num_ancilla_qubit, angle_sequence[0])
        projector_controller_gates | circuit([j for j in range(num_ancilla_qubit + 1)])
    return circuit