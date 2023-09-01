import math
import numpy as np
from QuICT.qcda.synthesis import QuantumStatePreparation
from QuICT.core.gate import CompositeGate, X, Z, CZ, CCZ, MultiControlToffoli
from QuICT.core import Circuit
from .unitary_matrix_encoding import *


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
        m_c_t = mct(m - 1)
        H | composite_gate(m - 1)
        m_c_t | composite_gate([i for i in range(m)])
        H | composite_gate(m - 1)
    x_composite_gate | composite_gate

    return composite_gate
##############################################################################################
# Following are function for truncation taylor expansion algorithm


def gates_R(num_reflection_qubit, num_qubit):
    """
    generate R = I - 2P gate
    """
    bit_string_array, _ = permute_bit_string(2**num_qubit - 1)
    R = CompositeGate()
    for i in range(2**num_reflection_qubit):
        reflection_gate = int_reflection(bit_string_array[i])
        reflection_gate | R
    return R


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
        temp_poly.append(np.log(2)**i / math.factorial(i))
    temp_poly = np.array(temp_poly)
    order = 0
    while np.sum(temp_poly) > error / times_steps:
        temp_poly = np.delete(temp_poly, 0, axis=0)
        order += 1
    order = order - 1
    return order


def calculate_expected_matrix(hamiltonian, time):
    eigenvalue, eigenbasis = np.linalg.eig(hamiltonian)
    matrix = np.exp(-1j * eigenvalue[0]) * np.kron(
        eigenbasis[0].reshape(len(eigenbasis[0]), 1), eigenbasis[0])
    for i in range(1, len(eigenvalue)):
        matrix = matrix + np.exp(-1j * eigenvalue[i] * time) * np.kron(
            eigenbasis[i].reshape(len(eigenbasis[i]), 1), eigenbasis[i])
    return matrix


def find_time_steps(coef_array, time):
    T = np.sum(coef_array) * time
    r = T / np.log(2)
    # use upper r
    time_steps = 0
    while time_steps < r:
        time_steps += 1
    return time_steps


def find_matrix_dimension(matrix_array):
    matrix_dimension = 0
    while 2 ** matrix_dimension != len(matrix_array[0][0]):
        matrix_dimension += 1
    return matrix_dimension


def calculate_approxiamate_matrix(hamiltonian, order, time, time_order):
    approximate_matrix = np.identity(len(hamiltonian[0]))
    for i in range(1, order):
        temp_matrix = hamiltonian
        for _ in range(i - 1):
            temp_matrix = np.matmul(temp_matrix, hamiltonian)
        temp_matrix = 1 / math.factorial(i) * (-1j * time / time_order)**i * temp_matrix
        approximate_matrix = np.sum((approximate_matrix, temp_matrix), axis=0)
    approximate_hamiltonian = approximate_matrix
    return approximate_hamiltonian


def TS_method(coefficient_array, matrix_array, time, error, max_order, initial_state):
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
     _) = read_unitary_matrix(coefficient_array, matrix_array)
    # step for bounding error
    #################################################################################
    assert check_hermitian(hamiltonian), "The hamiltonian is not hermitian."
    # calcualte time steps:
    time_steps = find_time_steps(coefficient_array, time)
    # calculate order based on the error given
    order = find_order(time_steps, error)
    # calculate matrix dimension
    matrix_dimension = find_matrix_dimension(matrix_array)
    # reshape coefficient array
    assert order < max_order, (f"The max order exceed {max_order}. "
                               f"Adjust max allowed taylor expansion order.")
    ###################################################################################
    # steps for making composite gates
    # calculate the product of (sum_{k} H_{k})^k from 0 to order
    coefficient_array, control_hamilton_list = product_gates(coefficient_array,
                                                             matrix_array,
                                                             order,
                                                             time,
                                                             time_steps)
    _, num_ancilla_qubit = permute_bit_string(
        len(coefficient_array) - 1)
    # make B gates, select V gates, ancilla reflection gates
    initial_state_gate = prepare_G_state(initial_state, np.sum(initial_state))
    print("Finding B gate")
    B = prepare_G_state(coefficient_array, np.sum(coefficient_array))
    B_dagger = B.inverse()
    print("Finding select-V gate")
    select_v, select_v_inverse = multicontrol_unitary(control_hamilton_list)
    # reflection gate
    print("Finding reflection gate")
    R = gates_R(matrix_dimension, num_ancilla_qubit +
                matrix_dimension + 1)  # +1ancilla qubit for control-v
    summed_coefficient = np.sum(coefficient_array)
    print("Truncate order:", order)
    print("Time steps:", time_steps)
    print("Summed coefficient:", summed_coefficient)
    print("Expected error:", np.abs(summed_coefficient - 2))
    print("Amplification size:", summed_coefficient / 2)
    print("Approximate time evolution operator:",
          calculate_approxiamate_matrix(hamiltonian, order, time, time_steps))
    ###########################################################################################
    # completing circuit
    print("Completing circuit")
    cg_width = num_ancilla_qubit + matrix_dimension + 1
    circuit = Circuit(cg_width)
    initial_state_gate | circuit([num_ancilla_qubit + 1 for i in range(matrix_dimension)])
    # W
    B | circuit
    select_v | circuit
    B_dagger | circuit
    # R
    R | circuit
    # W dagger
    B | circuit
    select_v_inverse | circuit
    B_dagger | circuit
    # R
    R | circuit
    # W
    B | circuit
    select_v | circuit
    B_dagger | circuit
    # -1
    R | circuit
    print("Circuit generation completes.")
    return circuit, cg_width, time_steps
