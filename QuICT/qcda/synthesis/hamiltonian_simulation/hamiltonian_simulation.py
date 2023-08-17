import math
from unitary_matrix_encoding import permute_bit_string, check_hamiltonian, UnitaryMatrixEncoding, product_gates, padding_coefficient_array
from quict_polynomial import Poly
from QuICT.qcda.synthesis import QuantumStatePreparation
from QuICT.core import Circuit
from QuICT.core.gate import *
import numpy as np
from quantum_signal_processing import QuantumSignalProcessing, SignalAngleFinder


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


def gates_R(num_qubit):
    """
    generate R = I - 2P gate
    """
    bit_string_array, _ = permute_bit_string(2**num_qubit-1)
    print(bit_string_array)
    R = CompositeGate()
    for binary_string in bit_string_array:
        reflection_gate = int_reflection(binary_string)
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
    order = 0
    range_min = 2-error/times_steps
    range_max = 2+error/times_steps
    s = 1/math.factorial(order)*np.log(2)**order

    while not (range_min < s and s < range_max):
        order += 1
        s += 1/math.factorial(order)*np.log(2)**order

    return order


class HamiltonianSimulation():
    def __init__(self):
        pass

    def TS_method(self, coefficient_array, matrix_array, time, times_steps, error: float = 0.001, max_order=20):
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
        (hamiltonian,
         coefficient_array,
         matrix_array,
         summed_coefficent) = check_hamiltonian(coefficient_array, matrix_array)

        matrix_dimension = 0
        while 2 ** matrix_dimension != len(matrix_array[0][0]):
            matrix_dimension += 1
        # calculate order based on the error given
        order = find_order(times_steps, error)
        assert order < max_order, f"The max order exceed {max_order}. Adjust max allowed taylor expansion order."
        # Algorithm
        coefficient_array, hamilton_list = product_gates(coefficient_array, matrix_array, order, time,
                                                         1)
        bit_string_array, num_ancilla_qubit = permute_bit_string(
            len(coefficient_array) - 1)
        # make B gates, select V gates, ancilla reflection gates
        UME = UnitaryMatrixEncoding()
        matrix_array = []
        matrix_array.append(hamilton_list[0])
        for i in range(1, len(hamilton_list)):
            temp_matrix = hamilton_list[i][0]
            for j in range(1, len(hamilton_list[i])):
                temp_matrix = np.matmul(temp_matrix, hamilton_list[i][j])
            matrix_array.append(temp_matrix)
        matrix_array = np.array(matrix_array)
        B, B_dagger, select_v = UME.execute(coefficient_array, matrix_array)
        R = gates_R(num_ancilla_qubit)

        circuit = Circuit(num_ancilla_qubit + matrix_dimension)
        # W
        B | circuit
        select_v | circuit
        B_dagger | circuit

        # R
        R | circuit

        # W dagger

        B | circuit
        select_v.inverse() | circuit
        B_dagger | circuit
        # R
        R | circuit
        # W
        B | circuit
        select_v | circuit
        B_dagger | circuit
        # -1

        whole_circuit_reflection = gates_R(
            num_ancilla_qubit + matrix_dimension)
        whole_circuit_reflection | circuit
        return circuit