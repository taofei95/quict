import math
from .unitary_matrix_encoding import *
from QuICT.qcda.synthesis import QuantumStatePreparation
from QuICT.core.gate import *
import numpy as np


def prepare_eigenvector_gate(input_eigenvector):
    """
    Given 2*n size state vector, find gates act on [1,0....., 0{n}] such that produce this vectir
    :param input_eigenvector: array of 1-D eigenvector
    :return: list of corresponded state preparation gates
    """
    gates_list = []
    for i in range(len(input_eigenvector)):
        qsp = QuantumStatePreparation('uniformly_gates')
        gates_list.append(qsp.execute(input_eigenvector[i]))
    return gates_list


def int_reflection(binary_string):
    """
    In the standard basis, reflect the matrix elements |binary_string><binary_string|
    generate control z composite gates.

    :param binary_string: input binary string.
    :return: Composite gate
    """
    m = len(binary_string)
    composite_gate = CompositeGate()
    x_composite_gate = CompositeGate()
    for i in range(len(binary_string)):
        if binary_string[i] == "0":
            X | x_composite_gate(i)
    x_composite_gate | composite_gate
    if m == 1:
        Z | composite_gate(0)
    elif m == 2:
        CZ | composite_gate([0, 1])
    elif m == 3:
        CCZ | composite_gate([0, 1, 2])
    elif m > 3:
        mct = MultiControlToffoli('no_aux')
        m_c_t = mct(m-1)
        H | composite_gate(m-1)
        m_c_t|composite_gate([i for i in range(m)])
        H | composite_gate(m-1)
    x_composite_gate | composite_gate

    return composite_gate
##############################################################################################
# Following are function for truncation taylor expansion algorithm


def gates_B(coefficient_array):
    """
    The B gate of equation 8
    """
    new_coefficient_array = padding_coefficient_array(coefficient_array)
    summed_coefficient = np.sum(new_coefficient_array)
    normalized_vector = np.sqrt(new_coefficient_array/summed_coefficient)
    B = prepare_eigenvector_gate(normalized_vector)
    B_dagger = B.inverse()
    return B, B_dagger


def gates_R(num_reflection_qubit,num_qubit):
    """
    generate R = I - 2P gate
    """
    bit_string_array, _ = permute_bit_string(2**num_qubit-1)
    R = CompositeGate()
    for i in range(2**num_reflection_qubit):
        reflection_gate = int_reflection(bit_string_array[i])
        reflection_gate | R
    return R


def find_order(summed_coefficient, times_steps, error):
    """
    Find the minimum order makes the |summed_coefficient-2|<=2
    :param times_steps: int
                        num of time steps
    :param error: float
                    accuracy of the algorithm
    :return: int
            minimum order of taylor expansion
    """
    order = 0
    range_min = 2-error/times_steps
    range_max = 2+error/times_steps
    s = 1/math.factorial(order)*np.log(summed_coefficient)**order

    while not (range_min < s and s < range_max):
        order += 1
        s += 1/math.factorial(order)*np.log(2)**order

    return order

def find_order(times_steps, error):
    """
    Find the minimum order makes the |summed_coefficient-2|<=2
    :param times_steps: int
                        num of time steps
    :param error: float
                    accuracy of the algorithm
    :return: int
            minimum order of taylor expansion
    """
    temp_poly = []
    for i in range(30):
        temp_poly.append(np.log(2)**i/math.factorial(i))
    temp_poly = np.array(temp_poly)
    order = 0
    while np.sum(temp_poly)>error/times_steps:
        temp_poly = np.delete(temp_poly,0,axis = 0)
        order+=1
    order = order-1
    return order
def calculate_expected_matrix(hamiltonian, time):
    eigenvalue, eigenbasis = np.linalg.eig(hamiltonian)
    matrix = np.exp(-1j*eigenvalue[0])*np.kron(eigenbasis[0].reshape(len(eigenbasis[0]),1), eigenbasis[0])
    for i in range(1, len(eigenvalue)):
        matrix = matrix + np.exp(-1j*eigenvalue[i]*time)*np.kron(eigenbasis[i].reshape(len(eigenbasis[i]),1), eigenbasis[i])
    return matrix
def calculate_approxiamate_matrix(hamiltonian, order, time, time_order):
    approximate_matrix = np.identity(len(hamiltonian[0]))
    for i in range(1, order):
        temp_matrix = hamiltonian
        for _ in range(i-1):
            temp_matrix = np.matmul(temp_matrix, hamiltonian)
        temp_matrix = 1/math.factorial(i)*(-1j*time/time_order)**i*temp_matrix
        approximate_matrix = np.sum((approximate_matrix, temp_matrix), axis=0)
    approximate_hamiltonian = approximate_matrix
    return approximate_hamiltonian

def TS_method(coefficient_array, matrix_array, time, error, max_order):
    """
    https://arxiv.org/abs/1412.4687
    Let hamiltonian satisfy:
    1. H = summed_{L}_{l=1}(coefficient_{l}*Unitary_{l})
    Compute truncation taylor series hamiltonian simulation algorithm
    :param coefficient_array: Array of coefficient in equation 1.

    :param matrix_array: Array of unitary matrix in equation 1.
    :param time: float

    :param times_steps: int
    :param error: float, Algorithm accuracy
    :param max_order: Maximum degree of taylor expansion allowed.
    :return: Quantum circuit
    """
    print("Algorithm start")
    (hamiltonian,
     coefficient_array,
     matrix_array,
     summed_coefficent) = read_unitary_matrix(coefficient_array, matrix_array)
    #step for bounding error
    #################################################################################
    #calcualte time steps:
    T = np.sum(coefficient_array)*time
    r = T/np.log(2)
    #use upper r
    time_steps = 0
    while time_steps<r:
        time_steps+=1
    assert check_hermitian(hamiltonian), "The hamiltonian is not hermitian."
    matrix_dimension = 0
    while 2 ** matrix_dimension != len(matrix_array[0][0]):
        matrix_dimension += 1
    # calculate order based on the error given
    order = find_order(time_steps, error)
    #reshape coefficient array
    assert order < max_order, f"The max order exceed {max_order}. Adjust max allowed taylor expansion order."
    ###################################################################################
    #steps for making composite gates
    #calculate the product of (sum_{k} H_{k})^k from 0 to order
    coefficient_array, control_hamilton_list = product_gates(coefficient_array, matrix_array, order, time,
                                                     time_steps)
    bit_string_array, num_ancilla_qubit = permute_bit_string(
        len(coefficient_array) - 1)
    # make B gates, select V gates, ancilla reflection gates
    print("Finding B gate")
    B = prepare_G_state(coefficient_array, np.sum(coefficient_array))
    B_dagger = B.inverse()
    print("Finding select-V gate")
    select_v, select_v_inverse = multicontrol_unitary(control_hamilton_list)
    #reflection gate
    print("Finding reflection gate")
    R = gates_R(matrix_dimension, num_ancilla_qubit+matrix_dimension+1)#+1ancilla qubit for control-v
    summed_coefficient = np.sum(coefficient_array)
    print("Truncate order:", order)
    print("Time steps:", time_steps)
    print("Summed coefficient:", summed_coefficient)
    print("Expected error:", np.abs(summed_coefficient-2))
    print("Amplification size:", summed_coefficient/2)
    print("Approximate time evolution operator:",
          calculate_approxiamate_matrix(hamiltonian, order, time, time_steps))
    ###########################################################################################
    #completing circuit
    print("Completing circuit")
    cg = CompositeGate(num_ancilla_qubit + matrix_dimension+1)
    # W
    B | cg
    select_v | cg
    B_dagger | cg
    # R
    R | cg
    # W dagger
    B | cg
    select_v_inverse | cg
    B_dagger | cg
    # R
    R | cg
    # W
    B | cg
    select_v | cg
    B_dagger | cg
    # -1
    whole_circuit_reflection = gates_R(
        num_ancilla_qubit + matrix_dimension+1, num_ancilla_qubit + matrix_dimension+1)
    whole_circuit_reflection | cg
    print("Circuit generation completes.")
    return cg
