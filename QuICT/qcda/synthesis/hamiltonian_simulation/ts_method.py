import math
import numpy as np
import scipy
import logging
from QuICT.core.gate import CompositeGate, X, Z, CZ, CCZ, MultiControlToffoli, H
from QuICT.core import Circuit
from .unitary_matrix_encoding import *


def int_reflection(binary_string: str):
    """
    Compute |binary><binary| reflection
    For example if |01><01| reflection, generate [[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
    Args:
        binary_string (str): binary representation of a positive integer.

    Returns:
        CompositeGate: A composite gate
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
        mct_ini = MultiControlToffoli('no_aux')
        mct = mct_ini(m - 1)
        H | composite_gate(m - 1)
        mct | composite_gate([i for i in range(m)])
        H | composite_gate(m - 1)
    x_composite_gate | composite_gate

    return composite_gate
##############################################################################################
# Following are function for truncation taylor expansion algorithm


def gates_R(num_reflection_qubit: int, num_qubit: int):
    """
    generate R = I - 2P gate

    Args:
    num_reflection_qubit (int): The selected number of qubit are reflected in there subspace.
    num_qubit (int): Total number of qubits.

    Returns:
        CompositeGate: R = I - 2P
    """
    assert num_qubit >= num_reflection_qubit, "The num qubits must greater or equal to num reflection qubit."
    bit_string_array, _ = permute_bit_string(2**num_qubit - 1)
    R = CompositeGate()
    for i in range(2**num_reflection_qubit):
        reflection_gate = int_reflection(bit_string_array[i])
        reflection_gate | R
    return R


def find_order(times_steps: int, error: float):
    """
    Find the minimum order makes the |summed_coefficient-2|<=2

    Args:
    times_steps (int):
        num of time steps
    error (float):
        accuracy of the algorithm

    Returns:
        int: minimum order of taylor expansion
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


def calculate_target_matrix(hamiltonian: np.ndarray, time: float):
    """
    calculate target time evolution matrix e^-iHt

    Args:
    hamiltonian (np.ndarray): A hermitian matrix.
    time (float): time setted.

    Returns:
        ComposteGate: A composite gate
    """
    assert check_hermitian(hamiltonian), "Hamiltonian is not hermitian"
    matrix = scipy.linalg.expm(-1j * hamiltonian * time)
    return matrix


def find_time_steps(coef_array: np.ndarray, time: float):
    """
    find the suitable time steps such that make summed coefficient close to 2.

    Args:
    coef_array (np.ndarray): A array of coefficient assign infront of matrix V. (U = SUM coef * V )
    time (float): evolution time

    Returns:
        int: time steps of evolution
    """
    T = np.sum(coef_array) * time
    r = T / np.log(2)
    # use upper r
    time_steps = math.ceil(r)
    return time_steps


def find_matrix_dimension(matrix_array: np.ndarray):
    """
    Find the matrix dimension.

    Args:
    matrix_array (np.ndarray): Array of matrix([matrix, matrix, matrix])

    Returns:
        np.ndarray: A positive integer.
    """
    matrix_dimension = int(np.log2(matrix_array.shape[1]))
    return matrix_dimension


def calculate_approximate_matrix(hamiltonian: np.ndarray, order: int, time: float, time_order: int):
    """
    calculate the approximated e^-iHt/r

    Args:
        hamiltonian (np.ndarray): A numpy 2D array (H)
        order (int): hamiltonian highest truncated order
        time (float): the evolution time(t)
        time_order (int): positive integer (r)

    Returns:
        np.ndarray: approximated hamiltonian
    """
    approximate_matrix = np.identity(hamiltonian.shape[0])
    for i in range(1, order):
        temp_matrix = hamiltonian
        for _ in range(i - 1):
            temp_matrix = np.matmul(temp_matrix, hamiltonian)
        temp_matrix = 1 / math.factorial(i) * (-1j * time / time_order)**i * temp_matrix
        approximate_matrix = np.sum((approximate_matrix, temp_matrix), axis=0)
    approximate_hamiltonian = approximate_matrix
    return approximate_hamiltonian


def truncate_series(coefficient_array: np.ndarray, matrix_array: np.ndarray,
                    time: float, error: float, max_order: int, initial_state: np.ndarray):
    """
    https://arxiv.org/abs/1412.4687
    Let hamiltonian satisfy:
    1. H = summed_{L}_{l=1}(coefficient_{l}*Unitary_{l})
    Compute truncation taylor series hamiltonian simulation algorithm
    Args:
        coefficient_array (np.ndarray): Array of coefficient in equation 1.
        matrix_array (np.ndarray): Array of unitary matrix in equation 1.
        time (float): The evolution time.
        error (float): Algorithm accuracy
        max_order (int): Maximum degree of taylor expansion allowed.
        initial_state (np.ndarray): The initial state

    Returns:
        Circuit: circuit compute e^{-iHt/r}.
        dict: A dictionary contain following information
        "circuit_width": c_width,
        "time_steps": time_steps,
        "order": order,
        "summed_coeffcient": summed_coefficient,
        "expected_error": expected_error,
        "amplification_size": amplification_size,
        "approximated_time_evolution_operator": approximate_time_evolution_operator
    """
    logging.info("truncation series algorithm start")
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
    _, num_ancilla_qubit = permute_bit_string(len(coefficient_array) - 1)
    # make B gates, select V gates, ancilla reflection gates
    initial_state_gate = prepare_G_state(initial_state, np.sum(initial_state))
    logging.debug("Find B gates")
    B = prepare_G_state(coefficient_array, np.sum(coefficient_array))
    logging.debug("Find B inverse gates")
    B_dagger = B.inverse()
    logging.debug("Find select v gates")
    select_v, select_v_inverse = multicontrol_unitary(control_hamilton_list)
    # reflection gate
    logging.debug("Find reflection oracles")
    R = gates_R(matrix_dimension, num_ancilla_qubit +
                matrix_dimension + 1)  # +1ancilla qubit for control-v
    summed_coefficient = np.sum(coefficient_array)
    expected_error = np.abs(summed_coefficient - 2)
    amplification_size = summed_coefficient / 2
    approximate_time_evolution_operator = calculate_approximate_matrix(hamiltonian, order, time, time_steps)
    ###########################################################################################
    # completing circuit
    logging.info("completing circuit")
    c_width = num_ancilla_qubit + matrix_dimension + 1
    circuit = Circuit(c_width)
    initial_state_gate | circuit(
        [num_ancilla_qubit + 1 + i for i in range(matrix_dimension)])
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
    logging.info("Truncation series algorithm completed")
    circuit_info_dictionary = {
        "circuit_width": c_width,
        "time_steps": time_steps,
        "order": order,
        "summed_coeffcient": summed_coefficient,
        "expected_error": expected_error,
        "amplification_size": amplification_size,
        "approximated_time_evolution_operator": approximate_time_evolution_operator
    }
    return circuit, circuit_info_dictionary
