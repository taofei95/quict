import numpy as np
from QuICT.core.gate import MultiControlToffoli, X, CompositeGate
from QuICT.qcda.synthesis import QuantumStatePreparation, UnitaryDecomposition, GateTransform
import itertools
##########################################
# Following code do stardard-form encoding of a linear combination of Unitaries
# or, if choose conj mode, do the [H, sqrt(I-H^2)]
#                                 [sqrt(I-H^2), H]
#encoding


def check_hermitian(input_matrix: np.ndarray):
    """
    Check if input matrix hermitian, # H = H*
    Args:
        input_matrix: A matrix to be hermitian

    Returns:
        Bool: true if hermitian, false if not hermitian
    """
    matrix_conjugate = np.transpose(np.conj(input_matrix))
    if np.allclose(matrix_conjugate, input_matrix):
        return True
    return False


def check_unitary(input_matrix: np.ndarray, tolerance: float=1e-9):
    """
    Check if input matrix unitary
    Args:
        input_matrix: A matrix to be unitary

    Returns:
        Bool: true if unitary, false if not unitary
    """
    matrix_transpose = np.transpose(np.conj(input_matrix))
    if (np.abs(np.matmul(input_matrix, matrix_transpose).trace() - len(input_matrix)) < tolerance).all():
        return True
    return False


def read_unitary_matrix(coefficient_array: np.ndarray, unitary_matrix_array: np.ndarray):
    """
    Read hamiltonian and check if input unitary matrix satisfies necassary conditions.
    1. elements in unitary matrix array must be unitary
    2. The whole matrix must be hermitian
    Args:
        coefficient_array: array of coefficient
        unitary_matrix_array: array of unitary matrix

    Returns:
        hamiltonian: hamiltonian = sum{i} coefficient_{i} unitary_{i}
        coefficient_array: input coefficient array but in complex 128 type
        unitary_matrix_array: input array but in complex 128 type
        summed_coefficient: summed cofficient array

    """
    unitary_matrix = unitary_matrix_array[0]
    assert len(unitary_matrix[:, 0]) == len(
        unitary_matrix[0]), "Not a square matrix"

    coefficient_array = coefficient_array.astype('complex128')
    unitary_matrix_array = unitary_matrix_array.astype('complex128')
    hamiltonian_array = []
    summed_coefficient = 0
    for i in range(len(coefficient_array)):
        hamiltonian_array.append(unitary_matrix_array[i] * coefficient_array[i])
        summed_coefficient = coefficient_array[i] + summed_coefficient
    hamiltonian_array = np.array(hamiltonian_array)
    hamiltonian = np.sum(hamiltonian_array, axis=0)
    return hamiltonian, coefficient_array, unitary_matrix_array, summed_coefficient


def padding_coefficient_array(coefficient_array: np.ndarray):
    """
    Padding a cofficient array with length m to 2**n if m<2**n
    Args:
        coefficient_array: array of coefficient
    Returns:
        coefficient_array: padded coefficient array
        n:Integer. coefficient array has length 2**n. Also represent the number of qubits required
            to represent this coefficient array.
    """
    length = len(coefficient_array)
    assert length != 0, f"The input coefficient_array can't has length {length}."
    n = 0
    while (2**n) < length:
        n += 1
    if n == 0 and length != 0:
        n = 1
    coefficient_array = np.pad(coefficient_array, (0, 2**n - length))
    return coefficient_array, n


def permute_bit_string(max_int: int):
    """
    Given max int calculate bit string from 0 to this num.
    Args:
        max_int: The maxmum int of returned bit string array
    Returns:
        array of bit string, minimum num of qubits need to express max int
    """
    # find max bound of max_int
    num_qubits = 0
    while 2 ** num_qubits - 1 < max_int:
        num_qubits += 1
    if max_int == 0:
        num_qubits = 1
    bit_string_array = np.arange(0, max_int, 1)
    permute_list = []
    for i in range(len(bit_string_array) + 1):
        permute_list.append(format(i, f"0{num_qubits}b"))
    return permute_list, num_qubits


def prepare_G_state(coefficient_array: np.ndarray, summed_coefficient: float):
    """
    Prepare |G> = sum_{i}(sqrt(coeffcient{i})/summed_coefficient* | i>)
    |i> in standard basis
    Args:
        coefficient_array: array of coefficient array
        summed_coefficient: sum(coefficient_array)
    Returns:
        composite gates who generate |G> after acting on |0>

    """
    state_vector = []
    # The coefficient array are in length 2**n.
    coefficient_array, _ = padding_coefficient_array(coefficient_array)
    for i in range(len(coefficient_array)):
        state_vector.append(
            np.sqrt(coefficient_array[i] / np.abs(summed_coefficient)))
    state_vector = np.sqrt(coefficient_array / summed_coefficient)
    QSP = QuantumStatePreparation('uniformly_gates', keep_phase=True)
    oracle_G = QSP.execute(state_vector)
    return oracle_G


def matrix_to_control_gate(matrix_array: np.ndarray, control_bit: int=1):
    """
    make a array of matrix to control matrix.
    if control bits set to 1. Then do the math |1><1| tensor matrix + |0><0| tensor I
    if control bits set to 0. Then do the math |0><0| tensor matrix + |1><1| tensor I
    Args:
        matrix_array: array of matrix array
        control_bit: 1 or 0
    Returns:
        control_gate_list: list of control gates
        matrix_dimension: Number of qubits that can represent the matrix
    """
    assert control_bit == 1 or control_bit == 0, "control bits must be 0 or 1"
    matrix_dimension = int(np.log2(len(matrix_array[0][0])))
    # prepare control gates gates
    identity_matrix = np.identity(
        2 ** matrix_dimension).astype('complex128')
    project_zero = np.array([[1, 0], [0, 0]], dtype='complex128')
    project_one = np.array([[0, 0], [0, 1]], dtype='complex128')
    UD = UnitaryDecomposition(include_phase_gate=True)
    GT = GateTransform(keep_phase=True)
    # Do unitary decomposition for sake of saving running time when computing matrix
    # or do simulation
    control_gate_list = []
    if control_bit == 1:
        for i in range(len(matrix_array)):
            unitary_matrix = np.kron(project_zero, identity_matrix) + np.kron(project_one, matrix_array[i])
            control_gate_list.append(GT.execute(UD.execute(unitary_matrix)[0]))
    elif control_bit == 0:
        for i in range(len(matrix_array)):
            unitary_matrix = np.kron(
                project_one, identity_matrix) + np.kron(project_zero, matrix_array[i])
            control_gate_list.append(GT.execute(UD.execute(unitary_matrix)[0]))
    return control_gate_list, matrix_dimension


def product_gates(coefficient_array: np.ndarray, matrix_array: np.ndarray, order: int, time: float, time_step: int):
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
    Args:
        coefficient_array: coefficient of linear combination unitary
        matrix_array: unitary matrix of linear combination unitary
        order: order of taylor expansion
        time: time evolution time
        time_step: steps of evolution

    Returns:
        coefficient_array: The coefficent array be processed by taylor expansion. Namely convert the coefficient
                            in equation 1 to equation 6 in paper 10.1103/PhysRevLett.114.090502
        gate list: The control gate list in equation 6.
    """
    # prepare control gates gates
    control_gates, matrix_dimension = matrix_to_control_gate(matrix_array)
    # prepare reflection and global phase gate
    global_phase_gate_positive = matrix_to_control_gate(
        [1j*np.identity(2**(matrix_dimension))])[0][0]
    global_phase_gate_negative = matrix_to_control_gate(
        [-1j * np.identity(2**(matrix_dimension))])[0][0]
    global_phase_gate_minus = matrix_to_control_gate(
        [-1 * np.identity(2**(matrix_dimension))])[0][0]
    global_phase_gate_identity = matrix_to_control_gate(
        [1 * np.identity(2**(matrix_dimension))])[0][0]

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
                temp_coefficient = temp_coefficient * \
                    coefficient_array[permute_array[i][j]]
                cg = stretch_gates(
                    control_gates[permute_array[i][length_permute_array - j - 1]], cg)
            if (-1j)**k == -1j:
                cg = stretch_gates(global_phase_gate_negative, cg)
            elif (-1j)**k == 1j:
                cg = stretch_gates(global_phase_gate_positive, cg)
            elif (-1j)**k == -1:
                cg = stretch_gates(global_phase_gate_minus, cg)

            gate_list.append(cg)

            coefficient_list.append(
                (((time / time_step) ** k) / np.math.factorial(k)) * temp_coefficient)

    coefficient_array = np.array(coefficient_list)
    return coefficient_array, gate_list


def multicontrol_unitary(unitary_gate_array: np.ndarray):
    """
    Find composite gates generates matrix = sum_{i} |i><i| tensor U_{i}
    Args:
        unitary_gate_array: array of control unitary
    Returns:
        control_gate_list: list of control gates
        composite_gate_inverse: inversed control gates
    """
    binary_string, num_ancilla_qubits = permute_bit_string(
        len(unitary_gate_array) - 1)
    # initialize multicontrol toffoli
    composite_gate = CompositeGate()
    composite_gate_inverse = CompositeGate()

    mct_ini = MultiControlToffoli('no_aux')
    num_control_bits = num_ancilla_qubits
    cnx = mct_ini(num_control_bits)

    if num_ancilla_qubits == 1:
        cnx = mct_ini(0)
    cnx_width = cnx.width()
    for i in range(len(unitary_gate_array)):
        def add_X():
            for j in range(len(binary_string[i])):
                if binary_string[i][j] == "0":
                    X | composite_gate(j)
                    X | composite_gate_inverse(j)
        add_X()
        if cnx_width != 1:
            cnx | composite_gate([k for k in range(num_control_bits + 1)])
            cnx | composite_gate_inverse(
                [k for k in range(num_control_bits + 1)])

        unitary_gate_array[i] | composite_gate(
            [cnx_width + k - 1 for k in range(unitary_gate_array[i].width())])
        unitary_gate_array[i].inverse() | composite_gate_inverse(
            [cnx_width + k - 1 for k in range(unitary_gate_array[i].width())])
        if cnx_width != 1:
            cnx | composite_gate([k for k in range(num_control_bits + 1)])
            cnx | composite_gate_inverse(
                [k for k in range(num_control_bits + 1)])
        add_X()
    return composite_gate, composite_gate_inverse


def conjugation_encoding(hamiltonian: np.ndarray):
    """
    Find the matrix [H, sqrt(I-H H)]
                  [sqrt(I-H H), -H]
    Args:
        Hamiltonian: Hamiltonian matrix
    Returns:
        encoding_gates: composite gates realize the unitary matrix
        circuit_size: Num of qubits use
    """
    eigenvalue, eigenvector = np.linalg.eig(hamiltonian)
    conj_matrix = []
    for i in range(len(eigenvalue)):
        transpose_eigenvector = eigenvector[i].reshape(len(eigenvector[i]), 1)
        conj_matrix.append(
            np.sqrt(1 - eigenvalue[i]**2) * np.kron(transpose_eigenvector, eigenvector[i]))
    conj_matrix = np.sum(np.array(conj_matrix), axis=0)
    temp_matrix = np.hstack((conj_matrix, -1 * hamiltonian))
    block_encoding_matrix = np.hstack((hamiltonian, conj_matrix))
    block_encoding_matrix = np.vstack((block_encoding_matrix, temp_matrix))
    UD = UnitaryDecomposition(include_phase_gate=True)
    encoding_gates, _ = UD.execute(block_encoding_matrix)
    circuit_size = int(np.log2(len(block_encoding_matrix[0])))

    return encoding_gates, circuit_size


class UnitaryMatrixEncoding:
    def __init__(self, method: str):
        self.method = method
        assert self.method == "LCU" or self.method == "conj", "Only LCU or conj method are provided."

    def execute(self, coefficient_array: np.ndarray, matrix_array: np.ndarray, complete: bool = False):
        """
        if LCU mode, generate circuit of equation 7 of paper https://arxiv.org/abs/2002.11649
        if CONJ mode, generate circuit that equivalent to matrix    [H, sqrt(I-H H)]
                                                                    [sqrt(I-H H), -H]
        by unitary decompisition
        Args:
            coefficient_array: array of coefficient in the linear combination of Hamiltonian
            matrix_array: array of unitary in the linear combination of Hamiltonian
        If LCU mode
        Returns:
            G:G|0> = |G> = sum_{i}(sqrt(coeffcient{i})/summed_coefficient* | i>)
            G_inverse: Inverse of G operator
            unitary_encoding: sum_{i} |i><i| tensor U_{i}
            unitary_encoding_inverse: Inverse of unitary encoding
        If CONJ mode:
        Returns:
            unitary_encoding: [H, sqrt(I-H H)]
                              [sqrt(I-H H), -H]
            circuit encoding the matrix shown above
            unitary_encoding_width: num qubit being used.
        """
        (hamiltonian,
         coefficient_array,
         unitary_matrix_array,
         summed_coefficient) = read_unitary_matrix(coefficient_array, matrix_array)
        if self.method == "LCU":
            G = prepare_G_state(coefficient_array, summed_coefficient)
            G_inverse = G.inverse()
            control_unitary_gates, _ = matrix_to_control_gate(unitary_matrix_array)
            unitary_encoding, unitary_encoding_inverse = multicontrol_unitary(control_unitary_gates)

            if complete:
                cg = CompositeGate()
                G | cg
                unitary_encoding | cg
                G_inverse | cg

                return cg
            elif not complete:
                return G, G_inverse, unitary_encoding, unitary_encoding_inverse
        if self.method == "conj":
            unitary_encoding, unitary_encoding_width = conjugation_encoding(
                hamiltonian)
            return unitary_encoding, unitary_encoding_width
