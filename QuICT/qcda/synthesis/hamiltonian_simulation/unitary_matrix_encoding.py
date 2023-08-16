import numpy as np
from QuICT.core.gate import *
from QuICT.qcda.synthesis import QuantumStatePreparation
import itertools
##########################################
#Following code do stardard-form encoding of a linear combination of Unitaries
def check_hermitian(input_matrix):
    """
    Check if input matrix hermitian
    """
    #H = H*
    matrix_conjugate = np.transpose(np.conj(input_matrix))
    if (matrix_conjugate==input_matrix).any():
        return True
    return False
def check_unitary(input_matrix):
    """
    Check if input matrix unitary
    """
    matrix_inverse = np.linalg.inv(input_matrix)
    if (input_matrix==matrix_inverse).any():
        return True
    return False
def check_hamiltonian(coefficient_array, unitary_matrix_array):
    """
    Read hamiltonian and check if input unitary matrix satisfies necassary conditions.
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

    for i in range(len(unitary_matrix_array)):

        assert check_unitary(unitary_matrix_array), f"The {i}th matrix is not unitary."
    coefficient_array = coefficient_array.astype('complex128')
    unitary_matrix_array = unitary_matrix_array.astype('complex128')
    hamiltonian_array = []
    summed_coefficient = 0
    for i in range(len(coefficient_array)):
        #print(unitary_matrix_array[i],coefficient_array[i])
        hamiltonian_array.append(unitary_matrix_array[i]*coefficient_array[i])
        summed_coefficient = coefficient_array[i] + summed_coefficient

    hamiltonian_array = np.array(hamiltonian_array)
    hamiltonian = np.sum(hamiltonian_array, axis = 0)
    assert check_hermitian(hamiltonian), "The hamiltonian is not hermitian."
    return hamiltonian, coefficient_array, unitary_matrix_array, summed_coefficient

def padding_coefficient_array(coefficient_array):
    length = len(coefficient_array)
    n = 0
    while (2**n-1)<length:
        n+=1
    coefficient_array = np.pad(coefficient_array,(0,2**n-length))
    return coefficient_array, n
def permute_bit_string(max_int):
    """
    Given max int calculate bit string from 0 to this num.

    :param max_int: The maxmum int of returned bit string array
    :return: array of bit string, minimum num of qubits need to express max int
    """
    # find max bound of max_int
    num_qubits = 0
    while 2 ** num_qubits-1 < max_int:
        num_qubits += 1
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
        state_vector.append(np.sqrt(coefficient_array[i]/np.abs(summed_coefficient)))
    state_vector = np.sqrt(coefficient_array/summed_coefficient)
    QSP = QuantumStatePreparation('uniformly_gates')
    oracle_G = QSP.execute(state_vector)
    return  oracle_G

def product_gates(coefficient_array: np.array, hamiltonian_array: np.array, order: int, time: float, time_step: int):
    """
    We permute the combination of matrix multiplication.

    The permutation execute in the following order
    1. If an input_array has following form A = np.array([matrix_1, matrix_2 ... ,matrix_n])
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
    we set the last term index into 1 and set the second last term index into 2(1+1)
    3. we repeat step 1 and step 2 to find all posible combinations


    Parameters
    ----------
    input_array : np.array
        The input_array contain all of information of hamiltonian
    order : int
        The maximum order that remain in the truncated taylor expansion of a hamiltonian operator.

    Returns
    -------
    np.array
        The permutated elements.
        From left to right [1...1, 1...2, ..., 1...k, ..., 1... 21, ..., k...k ]

    """
    matrix_list = []
    coefficient_list = []
    matrix_list.append(np.identity(len(hamiltonian_array[0][0])).astype('complex128'))
    coefficient_list.append(1)
    for k in range(1, order + 1):
        permute_array = list(itertools.product(range(len(hamiltonian_array)), repeat=k))
        for i in range(len(permute_array)):
            temp_matrix = []
            temp_coefficient = 1
            for j in range(len(permute_array[i])):
                temp_matrix.append(hamiltonian_array[permute_array[i][j]])
                temp_coefficient = temp_coefficient * coefficient_array[permute_array[i][j]]
            temp_matrix[0] = temp_matrix[0] * (-1j)**k
            ###################################
            #to save running time
            matrix_list.append(temp_matrix)
            ####################################
            coefficient_list.append((((time / time_step) ** k) / np.math.factorial(k)) * temp_coefficient)

    coefficient_array = np.array(coefficient_list)

    return coefficient_array, matrix_list

def multicontrol_unitary(unitary_matrix_array):
    """
    Find composite gates generates matrix = sum_{i} |i><i| tensor U_{i}

    :return: Composite gates

    """
    binary_string, num_ancilla_qubits = permute_bit_string(len(unitary_matrix_array)-1)

    matrix_dimension = 0
    while 2**matrix_dimension!=len(unitary_matrix_array[0][0]):
        matrix_dimension+=1

    composite_gate = CompositeGate()
    mct = MultiControlToffoli('no_aux')
    num_control_bits = num_ancilla_qubits - 1
    c_n_x = mct(num_control_bits)

    for i in range(len(unitary_matrix_array)):
        identity_matrix = np.identity(2 ** matrix_dimension).astype('complex128')
        project_zero = np.array([[1, 0], [0, 0]], dtype='complex128')
        project_one = np.array([[0, 0], [0, 1]], dtype='complex128')
        unitary_matrix = np.kron(project_zero, identity_matrix) + np.kron(project_one, unitary_matrix_array[i])
        unitary_gate = Unitary(unitary_matrix)



        if num_ancilla_qubits==1:
            if binary_string[i] == "0":
                unitary_gate | composite_gate([i for i in range(matrix_dimension+1)])
            elif binary_string[i] == "1":
                X | composite_gate(0)
                unitary_gate | composite_gate([i for i in range(matrix_dimension+1)])
                X | composite_gate(0)
        if num_ancilla_qubits==2:
            for j in range(len(binary_string[i])):
                if binary_string[i][j] == "1":
                    X | composite_gate(j)
            CX | composite_gate([0,1])
            unitary_gate | composite_gate([i+1 for i in range(matrix_dimension+1)])
            CX | composite_gate([0,1])
            for j in range(len(binary_string[i])):
                if binary_string[i][j] == "1":
                    X | composite_gate(j)
        if num_ancilla_qubits>2:

            for j in range(len(binary_string[i])):
                if binary_string[i][j] == "1":
                    X | composite_gate(j)

            c_n_x | composite_gate([i for i in range(num_control_bits+1)])
            unitary_gate | composite_gate([num_control_bits+i for i in range(matrix_dimension+1)])
            c_n_x | composite_gate([i for i in range(num_control_bits+1)])

            for j in range(len(binary_string[i])):
                if binary_string[i][j] == "1":
                    X | composite_gate(j)
    return composite_gate
class  UnitaryMatrixEncoding:
    def __init__(self):
        pass
    def execute(self, coefficient_array: np.ndarray, matrix_array:np.ndarray, complete: bool=False):
        (hamiltonian,
         coefficient_array,
         unitary_matrix_array,
         summed_coefficient) = check_hamiltonian(coefficient_array, matrix_array)
        G = prepare_G_state(coefficient_array, summed_coefficient)
        G_inverse = G.inverse()
        unitary_encoding = multicontrol_unitary(unitary_matrix_array)

        if complete:
            cg = CompositeGate()
            G|cg
            unitary_encoding|cg
            G_inverse|cg

            return   cg
        elif not complete:
            return G, G_inverse, unitary_encoding


