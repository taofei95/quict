import numpy as np
from QuICT.core.gate import *
from QuICT.qcda.synthesis import QuantumStatePreparation, UnitaryDecomposition
import itertools
##########################################
# Following code do stardard-form encoding of a linear combination of Unitaries


def check_hermitian(input_matrix):
    """
    Check if input matrix hermitian
    """
    # H = H*
    matrix_conjugate = np.transpose(np.conj(input_matrix))
    if (matrix_conjugate == input_matrix).any():
        return True
    return False


def check_unitary(input_matrix):
    """
    Check if input matrix unitary
    """
    matrix_inverse = np.linalg.inv(input_matrix)
    if (input_matrix == matrix_inverse).any():
        return True
    return False


def read_unitary_matrix(coefficient_array, unitary_matrix_array):
    """
    Read hamiltonian and check input u matrix satisfies necassary conditions
    1. elements in unitary matrix array must be unitary
    2. The whole matrix must be hermitian
    :param coefficient_array: array of coefficient
    :param unitary_matrix_array: array of unitary matrix
    :return:list
            [sum_{i}(coefficient[i]*unitary_matrix_array[i])
            ,coefficient_array
            ,matrix_array
            sum_{i}(coefficient[i])
            ]
    """
    assert check_unitary(unitary_matrix_array), f"The matrix is not unitary."
    coefficient_array = coefficient_array.astype('complex128')
    unitary_matrix_array = unitary_matrix_array.astype('complex128')
    hamiltonian_array = []
    summed_coefficient = 0
    for i in range(len(coefficient_array)):
        hamiltonian_array.append(unitary_matrix_array[i]*coefficient_array[i])
        summed_coefficient = coefficient_array[i] + summed_coefficient
    hamiltonian_array = np.array(hamiltonian_array)
    hamiltonian = np.sum(hamiltonian_array, axis=0)
    return hamiltonian, coefficient_array, \
        unitary_matrix_array, summed_coefficient


def padding_coefficient_array(coefficient_array):
    length = len(coefficient_array)
    assert length != 0, \
        f"The input coefficient_array can't has length {length}."
    n = 0
    while (2**n) < length:
        n += 1
    if n == 0 and length != 0:
        n = 1
    coefficient_array = np.pad(coefficient_array, (0, 2**n-length))
    return coefficient_array, n


def permute_bit_string(max_int):
    """
    Given max int calculate bit string from 0 to this num.

    :param max_int: The maxmum int of returned bit string array
    :return: array of bit string, minimum num of qubits to express max int
    """
    # find max bound of max_int
    num_qubits = 0
    while 2 ** num_qubits-1 < max_int:
        num_qubits += 1
    if max_int == 0:
        num_qubits = 1
    bit_string_array = np.arange(0, max_int, 1)
    permute_list = []
    for i in range(len(bit_string_array)+1):
        permute_list.append(format(i, f"0{num_qubits}b"))
    return permute_list, num_qubits


def prepare_G_state(coefficient_array, summed_coefficient):
    """
    Prepare |G> = sum_{i}(sqrt(coeffcient{i})/summed_coefficient* | i>)
    |i> in standard basis
    :return: composite gates who generate |G> after acting on |0>

    """
    state_vector = []
    # The coefficient array are in length 2**n.
    coefficient_array, _ = padding_coefficient_array(coefficient_array)
    for i in range(len(coefficient_array)):
        state_vector.append(
            np.sqrt(coefficient_array[i]/np.abs(summed_coefficient)))
    state_vector = np.sqrt(coefficient_array/summed_coefficient)
    QSP = QuantumStatePreparation('uniformly_gates')
    oracle_G = QSP.execute(state_vector)
    return oracle_G


def matrix_to_control_gate(matrix_array, control_bit=1):

    matrix_dimension = 0
    while 2**matrix_dimension != len(matrix_array[0][0]):
        matrix_dimension += 1
    # prepare control gates gates
    identity_matrix = np.identity(
        2 ** matrix_dimension).astype('complex128')
    project_zero = np.array([[1, 0], [0, 0]], dtype='complex128')
    project_one = np.array([[0, 0], [0, 1]], dtype='complex128')

    UD = UnitaryDecomposition()
    # unitary decomposition for saving running time when computing matrix
    # or do simulation
    control_gate_list = []
    if control_bit == 1:
        for i in range(len(matrix_array)):
            unitary_matrix = np.kron(project_zero, identity_matrix)
            + np.kron(project_one, matrix_array[i])
            control_gate_list.append(UD.execute(unitary_matrix)[0])
    elif control_bit == 0:
        for i in range(len(matrix_array)):
            unitary_matrix = np.kron(project_one, identity_matrix)
            + np.kron(project_zero, matrix_array[i])
            control_gate_list.append(UD.execute(unitary_matrix))
    return control_gate_list, matrix_dimension


def product_gates(coefficient_array: np.array, matrix_array,
                  order: int, time: float, time_step: int):
    """
    We permute the combination of matrix multiplication.

    The permutation execute in the following order
    1. If an input_array has the form A = np.array([matrix_1,...,matrix_n])
    We write this in short-hand [1,2,3,4,...,n]
        We want to calcualte A^k. This imply we want to permute
        [1..n][1..n][1..n]...[1..n]_{k_th}

       We multiply matrix in the order of
       1 * 1...1_{k_th}
       1 * 1...2_{k_th}
       .
       .
       .
       1 * 1...n_{k_th}
    2. after the last term reach maximum index of the input_array,
    set the last term index into 1 and the second last term index into 2(1+1)
    3. we repeat step 1 and step 2 to find all posible combinations


    Parameters
    ----------
    input_array : np.array
        The input_array contain all of information of hamiltonian
    order : int
        The max order that remain in truncated tayl expan. of a hmtn operator.

    Returns
    -------
    np.array
        The permutated elements.
        From left to right [1..1, 1..2, .., 1..k, .., 1.. 21, .., k..k ]

    """
    # prepare control gates gates
    control_gates, matrix_dimension = matrix_to_control_gate(matrix_array)
    # prepare reflection and global phase gate

    global_phase_gate_positive = matrix_to_control_gate(
        [1j*np.identity(2**(matrix_dimension))])[0][0]
    global_phase_gate_negative = matrix_to_control_gate(
        [-1j*np.identity(2**(matrix_dimension))])[0][0]
    global_phase_gate_minus = matrix_to_control_gate(
        [-1*np.identity(2**(matrix_dimension))])[0][0]
    global_phase_gate_identity = matrix_to_control_gate(
        [1*np.identity(2**(matrix_dimension))])[0][0]

    # generate control gate list
    gate_list = []
    coefficient_list = []
    # add taylor expansion first order term
    gate_list.append(global_phase_gate_identity)
    coefficient_list.append(1)

    def stretch_gates(input_gates, target_composite_gates):
        decomposed_input_gates = input_gates.gate_decomposition()
        for m in range(len(decomposed_input_gates)):
            decomposed_input_gates[m][0] | target_composite_gates(
                decomposed_input_gates[m][1])
        return target_composite_gates
    # add taylor expansion higher order term
    for k in range(1, order + 1):
        permute_array = list(itertools.product(
            range(len(control_gates)), repeat=k))
        for i in range(len(permute_array)):
            cg = CompositeGate()
            temp_coefficient = 1
            length_permute_array = len(permute_array[i])
            for j in range(length_permute_array):
                temp_coefficient = temp_coefficient\
                        * coefficient_array[permute_array[i][j]]
                cg = stretch_gates(control_gates
                        [permute_array[i] [length_permute_array-j-1]], cg)
            if (-1j)**k == -1j:
                cg = stretch_gates(global_phase_gate_negative, cg)
            elif (-1j)**k == 1j:
                cg = stretch_gates(global_phase_gate_positive, cg)
            elif (-1j)**k == -1:
                cg = stretch_gates(global_phase_gate_minus, cg)

            gate_list.append(cg)

            coefficient_list.append(
                (((time / time_step) ** k) / np.math.factorial(k))
                * temp_coefficient)

    coefficient_array = np.array(coefficient_list)

    return coefficient_array, gate_list


def multicontrol_unitary(unitary_gate_array):
    """
    Find composite gates generates matrix = sum_{i} |i><i| tensor U_{i}

    :return: Composite gates

    """
    binary_string, num_ancilla_qubits = permute_bit_string(
        len(unitary_gate_array)-1)
    # initialize multicontrol toffoli
    composite_gate = CompositeGate()
    composite_gate_inverse = CompositeGate()

    mct = MultiControlToffoli('no_aux')
    num_control_bits = num_ancilla_qubits
    c_n_x = mct(num_control_bits)

    if num_ancilla_qubits == 1:
        c_n_x = mct(0)
    c_n_x_width = c_n_x.width()
    for i in range(len(unitary_gate_array)):
        def add_X():
            for j in range(len(binary_string[i])):
                if binary_string[i][j] == "0":
                    X | composite_gate(j)
                    X | composite_gate_inverse(j)

        add_X()
        if c_n_x_width != 1:
            c_n_x | composite_gate([k for k in range(num_control_bits+1)])
            c_n_x | composite_gate_inverse(
                [k for k in range(num_control_bits+1)])

        unitary_gate_array[i] | composite_gate(
            [c_n_x_width+k-1 for k in range(unitary_gate_array[i].width())])
        unitary_gate_array[i].inverse() | composite_gate_inverse(
            [c_n_x_width+k-1 for k in range(unitary_gate_array[i].width())])
        if c_n_x_width != 1:
            c_n_x | composite_gate([k for k in range(num_control_bits + 1)])
            c_n_x | composite_gate_inverse(
                [k for k in range(num_control_bits + 1)])
        add_X()
    return composite_gate, composite_gate_inverse


class UnitaryMatrixEncoding:
    def __init__(self):
        pass

    def execute(self, coefficient_array: np.ndarray,
                matrix_array: np.ndarray, complete: bool = False):
        (hamiltonian,
         coefficient_array,
         unitary_matrix_array,
         summed_coefficient) = read_unitary_matrix
        (coefficient_array, matrix_array)
        G = prepare_G_state(coefficient_array, summed_coefficient)
        G_inverse = G.inverse()
        unitary_encoding, unitary_encoding_inverse = \
            multicontrol_unitary(unitary_matrix_array)

        if complete:
            cg = CompositeGate()
            G | cg
            unitary_encoding | cg
            G_inverse | cg

            return cg
        elif not complete:
            return G, G_inverse, unitary_encoding, unitary_encoding_inverse
