from unitary_matrix_encoding import permute_bit_string,check_hamiltonian, UnitaryMatrixEncoding, product_gates, padding_coefficient_array
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
    return gates_list
def control_z(binary_string):
    m = len(binary_string)
    composite_gate = CompositeGate()
    if m == 2:
        for i in range(len(binary_string)):
            if binary_string[i]=="0":
                X|composite_gate(i)
        CZ|composite_gate([0,1])
        for i in range(len(binary_string)):
            if binary_string[i]=="0":
                X|composite_gate(i)


        return composite_gate
    elif m == 3:
        for i in range(len(binary_string)):
            if binary_string[i]=="0":
                X|composite_gate(i)
        CCZ|composite_gate([0,1,2])
        for i in range(len(binary_string)):
            if binary_string[i]=="0":
                X|composite_gate(i)
        return composite_gate
    elif m > 3:
        for i in range(m):
            if int(binary_string[i]) == 0:
                X | composite_gate(i)

        my_ccx = CCX & [0, 1, m]
        my_ccx | composite_gate
        for i in range(2, m):
            my_ccx = CCX & [i, i + m - 2, i + m - 1]
            my_ccx | composite_gate

        H | composite_gate(2*m-1)
        CX | composite_gate([2*m-2, 2*m-1])
        H | composite_gate(2*m-1)

        for i in range(0, m - 2):
            my_ccx = CCX & [m - i - 1, 2 * (m - 1) - i - 1, 2 * (m - 1) - i]
            my_ccx | composite_gate
        my_ccx = CCX & [0, 1, m]
        my_ccx | composite_gate

        for i in range(m):
            if int(binary_string[i]) == 0:
                X | composite_gate(i)
        return composite_gate
##############################################################################################
#Following are function for truncation taylor expansion algorithm

def gates_B(coefficient_array):
    new_coefficient_array = padding_coefficient_array(coefficient_array)
    summed_coefficient = np.sum(new_coefficient_array)
    normalized_vector = np.sqrt(new_coefficient_array/summed_coefficient)
    B = prepare_eigenvector_gate(normalized_vector)
    B_dagger = B.inverse()
    return B, B_dagger
def gates_R(num_ancilla_qubit):
    bit_string_array, num_ancilla_qubit = permute_bit_string(2**num_ancilla_qubit-1)
    R = CompositeGate()
    for binary_string in bit_string_array:
        reflection_gate = control_z(bit_string_array)
        reflection_gate | R
    return R









class HamiltonianSimulation():
    def __init__(self, method):
        self.method = method
        assert (method!="QSP" or method!="TS"), "Method must be quantum_signal_processing or truncation."
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
###########################################step in algorithm
        if self.method == "TS":
            bit_string_array, num_ancilla_qubit = permute_bit_string(max_ancilla_integer)
            coefficient_array, hamilton_list = product_algorithm(coefficient_array, hamiltonian_array, order, time,
                                                                 time_step)
        #make B gates, select V gates, ancilla reflection gates
            UME = UnitaryMatrixEncoding()
            matrix_array = []
            for i in range(len(hamilton_list)):
                temp_matrix = hamilton_list[i][0]
                for j in range(1,len(hamilton_list[i])):
                    temp_matrix = np.kron(temp_matrix, hamilton_list[j])
                matrix_array.append(temp_matrix)
            matrix_array = np.array(matrix_array)
            B, B_dagger, select_v = UME.execute(coefficient_array, matrix_array)

            R = gates_R(num_ancilla_qubit)

            circuit = Circuit(num_ancilla_qubit+matrix_dimension-1)
            #W
            B|circuit
            select_v | circuit(0)
            B_dagger | circuit(0)

            #R
            R | circuit(0)

            #W dagger

            B|circuit(0)
            select_v.inverse() | circuit(0)
            B_dagger|circuit(0)
            #R
            R |circuit(0)
            #W
            B|circuit
            select_v | circuit(0)
            B_dagger | circuit(0)
            #-1

            whole_circuit_reflection = gates_R(num_ancilla_qubit+matrix_dimension-1)
            whole_circuit_reflection | circuit(0)
            return circuit




        if self.method=="QSP":
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