from unitary_matrix_encoding import check_hamiltonian, UnitaryMatrixEncoding
from quict_polynomial import Poly
from QuICT.qcda.synthesis import QuantumStatePreparation
from QuICT.core import Circuit
from QuICT.core.gate import *
import numpy as np

from quantum_signal_processing import QuantumSignalProcessing, SignalAngleFinder
def prepare_eigenvector_gate(input_eigenvector):
    gates_list = []
    for i in range(len(input_eigenvector)):
        qsp = QuantumStatePreparation('uniformly_gates')
        gates.append(qsp.execute(input_eigenvector[i]))
    return gaets_list



class HamiltonianSimulation():
    def __init__(self, method):
        self.method = method
        assert (method!="quantum_signal_processing" or method!="truncation"), "Method must be quantum_signal_processing or truncation."
    def execute(self,time, coefficient_array, matrix_array):
        (hamiltonian,
         coefficient_array,
         matrix_array,
         summed_coefficent) = check_hamiltonian(coefficient_array, matrix_array)

        matrix_dimension = 0
        while 2 ** matrix_dimension != len(unitary_matrix_array[0][0]):
            matrix_dimension += 1

        length = len(coefficient_array)
        num_ancilla_qubit = 0
        while 2 ** num_ancilla_qubit != length:
            num_ancilla_qubit += 1

        eigenvalues, eigenvector = np.linalg.eig(hamiltonian)

        gates_list = prepare_eigenvector_gate(eigenvector)

        #signal processing
        #find poly
        exp_normal_basis = Poly()
        P = exp_normal_basis.normal_basis(time, 30)
        #find angle
        saf = SignalAngleFinder()
        angle_array = saf.execute(P, 30)
        #find circuit
        signal_circuit_list = []
        qsp = QuantumSignalProcessing(angle_array)
        for eigenvalue in range(eigenvalues):
            signal_circuit = qsp.signal_circuit(eigenvalue)
            signal_circuit_list.append(signal_circuit)
        #################################################################
        #quantum signal processing circuit
        circuit = Circuit()

        UME =  UnitaryMatrixEncoding()
        G, G_dagger,unitary_encoding=UME.execute(coefficient_array, matrix_array)
        G|circuit(0)
        unitary_encoding | circuit(0)
        H|circuit(num_ancilla_qubit+matrix_dimension)

        mct = MultiControlToffoli('no_aux')
        m_c_t = mct(matrix_dimension)

        for i in range(len(gates_list)):
            eigenvalue_gates = gates_list[i]
            eigenvalue_gates_dagger = eigenvalue_gates.inverse()
            eigenvalue_gates_dagger|circuit([i+num_ancilla_qubit for i in range(matrix_dimension)])
            for j in range(matrix_dimension):
                X|circuit(num_ancilla_qubit+j)
            m_c_t | circuit([i+num_ancilla_qubit for i in range(matrix_dimension+1)])
            signal_circuit_list[i]|circuit(num_ancilla_qubit+matrix_dimension)
            m_c_t|  circuit([i+num_ancilla_qubit for i in range(matrix_dimension+1)])
            for j in range(matrix_dimension):
                X|circuit(num_ancilla_qubit+j)
            eigenvalue_gates | circuit([i+num_ancilla_qubit for i in range(matrix_dimension)])
        G_dagger | circuit(0)
        unitary_encoding.inverse() | circuit(0)

        return circuit









