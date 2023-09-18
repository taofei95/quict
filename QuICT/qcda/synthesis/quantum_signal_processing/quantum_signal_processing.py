import numpy as np
from numpy.polynomial.polynomial import Polynomial
from QuICT.core.gate import H, Z, X, Rz, Rx, GPhase, CompositeGate, MultiControlToffoli
from QuICT.core import Circuit
from QuICT.qcda.synthesis.hamiltonian_simulation.unitary_matrix_encoding import UnitaryMatrixEncoding
#######################################################
"""
In this file, we achieve this motivation.
given an input x in the domain[-1,1], and polynomial p(x)
We find the SIGNAL MATRIX
[p(x), iQ(x) * sqrt(1-x^2)]
[i * Q*(x) * sqrt(1-x^2), P*(x)]

Define a sequence of angle phi = {phi_0, phi_1..... phi_k} include in R_{k+1}
This matrix is yield by computing sequence of matrix
e^(i * phi_{0} * pauli_z) * product{k}_{j=1} [ W(x) * e^i * phi_{j}_pauli_z]

That means if we know the sequence of angle, we can block encode x and polynomial p(x) in

This file provide following functionalities.
Given x and P(x)
1. find out Q(x), the conjugation of P
2. find out angle_sequence
3. Compute signal matrix and generate signal circuit

The symbol used in the code is consistent with the symbol that we introduced above.
"""


class SignalAngleFinder:
    def __init__(self, tolerance=1e-6):
        """
        
        Args:
            tolerance (float): The maximum deviation between input polynomial P and signal processing 
                       simulated polynomial
        """
        self.tolerance = tolerance

    def execute(self, polynomial_p: Polynomial, k: int):
        """
        Based on algorithm in paper https://arxiv.org/abs/1806.01838
        Given polynomial p find sequence of phi.
        Args:
           polynomial_p (Polynomial): P(x)
           k (int): phase angle length
        Returns:
            list[float]: phase angle in equaiton 3
        """
        polynomial_q = generate_poly_q(polynomial_p, k)
        phase_angle = generate_phase_angle(polynomial_p, polynomial_q, k)
        bool_function, error = complete_check(
            phase_angle, polynomial_p, self.tolerance)
        assert bool_function, f"The error:{error} exceed the maximum tolerance:{self.tolerance}."
        return phase_angle


class QuantumSignalProcessing:
    """
    Given x and angle_sequence
    find signal matrix and signal circuit
    """

    def __init__(self, angle_sequence: list[float]):
        """
        Args:
            angle_sequence (list[float]): The angle sequence shapes polynomial 
        """
        self.angle_sequence = angle_sequence

    def signal_matrix(self, input_x: float):
        """
        generate the circuit in equation A7 from paper:
        https://arxiv.org/abs/2105.02859
        Args:
           input_x (float): x of P(x)
        Returns:
            np.ndarray: Expected matrix that encoding P(x) in the top left hand corner.
        """
        WX = W_X_marix(input_x)
        signal_matrix = exp_pauli_z_matrix(self.angle_sequence[0])
        for i in range(1, len(self.angle_sequence)):
            signal_matrix = np.matmul(signal_matrix, np.matmul(WX, exp_pauli_z_matrix(self.angle_sequence[i])))
        return signal_matrix

    def signal_circuit(self, input_x: float):
        """
        generate the circuit in equation A7 from paper:
        https://arxiv.org/abs/2105.02859
        Args:
           input_x (float): x of P(x)
        Returns:
            QuICT.core.gate.CompositeGate
        """
        signal_x = -2 * np.arccos(input_x)
        signal_composite_gate = CompositeGate()
        WX = Rx(signal_x)
        signal_gates = Rz(-2 * self.angle_sequence[0])
        signal_gates | signal_composite_gate(0)
        for i in range(1, len(self.angle_sequence)):
            WX | signal_composite_gate(0)
            signal_gates = Rz(-2 * self.angle_sequence[i])
            signal_gates | signal_composite_gate(0)
        return signal_composite_gate

    def signal_processing_circuit(self, coefficient_array, matrix_array):
        """
        generate the circuit depicts in the figure 16 of paper;
        https://arxiv.org/abs/2002.11649
        This is the generalized circuit that use one less ancilla qubit
        Args:
           coefficient_array (np.array): array of coefficient of linear combination  of hamiltonian.
           matrix_array (np.ndarray): array of matrix of linear combination of hamiltonian.
        Returns:
            Circuit: QSP circuit
        """
        matrix_dimension = int(np.log2(len(matrix_array[0][0])))
        UME = UnitaryMatrixEncoding("LCU")
        U = UME.execute(coefficient_array, matrix_array, complete=True, phase_gate=False)
        circuit_width = U.width()
        cir = Circuit(circuit_width + 1)
        k = len(self.angle_sequence)
        H | cir(0)
        for i in range(k - 1):
            if i != 0:
                Z | cir(0)
            projector_controller(
                int(circuit_width - matrix_dimension), self.angle_sequence[-i - 1]) | cir
            U | cir([i + 1 for i in range(circuit_width)])
        Z | cir(0)
        projector_controller(
            int(circuit_width - matrix_dimension), self.angle_sequence[0]) | cir
        H | cir(0)
        GPhase(-np.pi/2-np.pi*((k/2)%4)) | cir(0)
        return cir


def check_poly_parity(poly_p: Polynomial, k: int, tolerance: float = 1e-6):
    """
    Check Theorem 3 condition 2.
    Args:
       poly_p (Polynomial): polynomial p
       k (int): Input order
       tolerance (float): treat residue as zero.
    Returns:
        bool: true if poly parity satisfy condition
    """
    # We define even parity as the coefficient of a polynomial associate with even order to be non-zero.
    parity = k % 2
    poly_p_coefficient = poly_p.coef
    # poly_p must have this parity. So all terms correspond to another parity(if even parity then odd term)
    # must be zero
    # if poly p doesn't have this parity return flase
    if np.max(np.abs(poly_p_coefficient[np.abs(parity - 1)::2])) > tolerance:
        return False
    return True


def W_X_marix(input_signal: float):
    """
    Args:
        input_signal (float): input x
    Returns:
        np.ndarray: [x, i*sqrt(1-x^2)]
                    [isqrt(1-x^2), x]
    """
    assert (-1 <= input_signal <= 1), "input_signal must be in domain[-1,1]"
    return np.array([[input_signal, np.sqrt(1 - input_signal**2) * 1j],
                     [np.sqrt(1 - input_signal**2) * 1j, input_signal]])


def exp_pauli_z_matrix(angle_phi: float):
    """
    generate e^(i*phi_{i} * pauli_z)
    Args:
        angle_phi (float): Phi in the set{[phi_0, phi_1, phi_2...phi_k}
    Returns:
        np.ndarray: [e^i*angle_phi, 0]
                    [0, e^-i*angle_phi]
    """
    return np.array([[np.exp(angle_phi * 1j), 0],
                     [0, np.exp(angle_phi * (-1j))]])


def block_encoding_matrix(signal_x: float, polynomial_p: Polynomial, polynomial_q: Polynomial):
    """
    Generate matrix [p(x), iQ(x) * sqrt(1-x^2)]
                    [i*Q*(x)*sqrt(1-x^2), P*(x)]
    Args:
        signal_x (float): As the convention
        polynomial_p (Polynomial): P(x)
        polynomial_q (Polynomial): conjugation to P.
    Returns:
        np.ndarray:
        [p(x), iQ(x) * sqrt(1-x^2)]
        [i*Q*(x)*sqrt(1-x^2), P*(x)]
    """
    polynomial_q_conjugate = Polynomial(np.conj(polynomial_q.coef))
    polynomial_p_conjugate = Polynomial(np.conj(polynomial_p.ceof))
    return np.array([[polynomial_p(signal_x), 1j * polynomial_q(signal_x) * np.sqrt(1 - signal_x**2)],
                    [1j * polynomial_q_conjugate(signal_x) * np.sqrt(1 - signal_x**2), polynomial_p_conjugate(signal_x)]])


def check_polynomial_normalization(p: Polynomial, q: Polynomial, steps: int = 10, tolerance: float = 1e-4):
    """
     Check theorem condition 3.
    Args:
        p (Polynomial): P(x)
        q (Polynomial): conjugation to P.
        steps (int): num of steps in range [-1, 1].
        tolerance (float): Max deviation from 1.
    Returns:
        bool: True if Theorem 3 satisfied, false if not satisfied.
    """
    summed_poly = p * Polynomial(np.conj(p.coef)) + Polynomial([1, 0, -1]) * q * Polynomial(np.conj(q.coef))
    x = np.linspace(-1, 1, steps)
    max_deviation = np.max(np.abs(np.abs(summed_poly(x)) - 1))
    if max_deviation > tolerance:
        return False, max_deviation
    return True, max_deviation


def complete_check(phi_list: np.ndarray, poly_p: Polynomial, tolerance: float=1e-6):
    """
    Check if the p(x) generated by signal processing which encoded in top left
     hand side of signal matrix is same as the input P(x).
    Arg:
        phi_list (np.ndarray): phi sequence
        poly_p (Polynomial): P(x)
        tolerance (float): Max deviation from p(x)
    Returns:
        bool: True or False based on condition
        largest_error: The max deviation from 0.
    """
    # check if equation 3 give rise to the polynomial p within the maximum tolerance.
    x = np.linspace(-1, 1, 10)
    target_poly_p = np.array(list(map(poly_p, x)))
    rz_sequence = list(map(exp_pauli_z_matrix, phi_list))
    WX = list(map(W_X_marix, x))
    calculated_poly_p = []
    for i in range(len(WX)):
        temp = rz_sequence[0]
        for j in range(1, len(rz_sequence)):
            temp = np.matmul(temp, np.matmul(WX[i], rz_sequence[j]))
        calculated_poly_p.append(temp[0][0])

    largest_error = np.max(np.abs(target_poly_p - np.array(calculated_poly_p)))
    if largest_error > tolerance:
        return False, largest_error
    elif largest_error < tolerance:
        return True, largest_error


#######################################################
# following codes help generate polynomial Q who is the conjugate of polynomial p.
# The algorithm comes from Theorem 4.
def generate_A_tilt(polynomial_p: Polynomial):
    """
    A = 1-P(x)*P*(x). A is even polynomial. Set y=x^2
    A_tilt = A(y) = A(x^2)
    arg:
        polynomial_p (Polynomial): The polynomial P.
    return:
        Polynomial: A(y)
        float: The leading coefficient of A(y)
    """
    polynomial_p_conjugate = Polynomial(np.conj(polynomial_p.coef))
    A = 1 - polynomial_p * polynomial_p_conjugate
    A_coefficient = A.coef
    A_tilt = Polynomial(A_coefficient[0::2])
    A_tilt_leading_coefficient = A_tilt.coef[A_tilt.degree()]
    return A_tilt, A_tilt_leading_coefficient


def generate_A_tilt_root(A_tilt: Polynomial, tolerance: float = 1e-3):
    """
    Args:
    A_tilt (Polynomial): A(y)
    tolerance (float): All value below tolerance is treated as non significant roots

    return:
        np.array: P(x) roots. Noting that complex roots comes in pairs and according to the theorem 4,
                We only need one roots in the complex multisets.
    """
    ######################################################
    #TODO, Find a more stable method to solve roots
    ######################################################
    # All real root have even multiplicity except for one.
    # To avoid degeneracy root, we remove zero as well
    # get rid of 0 and 1.
    temp_roots = np.array([root for root in A_tilt.roots() if not ((np.abs(np.real(root) - 1) < tolerance
                                                                    and np.abs(np.imag(root) < tolerance))
                                                                   or np.abs(root) < tolerance)])
    #To account the complex roots and real roots
    for i in range(len(temp_roots)):
        for j in range(len(temp_roots)):
            if (np.abs(np.real(temp_roots[i]) - np.real(temp_roots[j])) < tolerance) and i != j:
                temp_roots = np.delete(temp_roots, j)
                break
        if len(temp_roots) == i + 1:
            break
    return temp_roots


def generate_poly_q(polynomial_p: Polynomial, k: int):
    """
    Do calculation in equation 7 and equation 8.
    Args:
    polynomial_p (Polynomial): P(x)
    k (int): Order of polynomial P.

    return:
     Polynomial: Q.
    """
    #################
    #TODO, find a stable method to generate polynomial Q. See paper https://arxiv.org/abs/2002.11649
    #################
    assert polynomial_p.degree() <= k, "Poly P order must larger than poly Q order."
    assert check_poly_parity(polynomial_p, k), "Poly P parity doens't satisfy the condition"
    A_tilt, A_tilt_leading_coefficient = generate_A_tilt(polynomial_p)
    A_tilt_roots = generate_A_tilt_root(A_tilt)
    # algorithm find Q
    polynomial_q = Polynomial([np.sqrt(-1 * A_tilt_leading_coefficient)])
    for roots in A_tilt_roots:
        polynomial_q = polynomial_q * Polynomial([-1 * roots, 0, 1])
    if k % 2 == 0:
        polynomial_q = polynomial_q * Polynomial([0, 1])
    return polynomial_q
###########################################################################################


def update_polynomial(phi: float, polynomial_p: Polynomial, polynomial_q: Polynomial):
    """
    Do calculation in equation 7 and equation 8.
    Args:
        polynomial_p (Polynomial): P(x)
        polynomial_q (Polynomial): Q(x)
        phi (float): angle from angle sequence

    return:
        Polynomial: updated p and q.
    """
    # The input polynomial must have odd or odd parity.
    new_poly_p = (np.exp(-1j * phi) * Polynomial([0, 1]) * polynomial_p +
                  np.exp(1j * phi) * Polynomial([1, 0, -1]) * polynomial_q)
    new_poly_q = (np.exp(1j * phi) * Polynomial([0, 1])
                  * polynomial_q - np.exp(-1j * phi) * polynomial_p)
    # highest order of poly p and poly q must be canceled out,
    # however, the numerical unstability can't gaurantee the gradual reduced order during deduction hence need
    # manually remove last two terms of new poly p and new poly q.
    if new_poly_p.degree() >= polynomial_p.degree():
        new_poly_p.coef = np.delete(np.delete(new_poly_p.coef, -1), -1)
    if new_poly_q.degree() >= polynomial_q.degree():
        new_poly_q.coef = np.delete(np.delete(new_poly_q.coef, -1), -1)
    return new_poly_p, new_poly_q


def generate_phase_angle(polynomial_p: Polynomial, polynomial_q: Polynomial, k: int):
    """
    Algorithm of theorem 3.
    calculate sequence of phi
    e^(2*i*phi) = p_l/q_{l-1}
    p_l is the last non zero term of polynomial p and q_l-1 is second last non zero term of polynomial q.

    Args:
    Polynomial_p (Polynomial): P(x)
    polynomial_q (Polynomial): Q(x)

    Returns:
        np.array: phase angle.
    """
    degree_p = polynomial_p.degree()
    degree_q = polynomial_q.degree()
    # condition checks
    bool_function, deviation = check_polynomial_normalization(polynomial_p, polynomial_q)
    # check Theorem 3.1
    assert degree_p <= k, "The degree of polynomial p must lower or equal to k."
    assert degree_q <= max(0, k - 1), "The degree of polynomial q must lower or equal to k-1."
    # check Theorem 3.2
    assert check_poly_parity(polynomial_p, k), "The parity of polynomial p is not satisfied."
    assert check_poly_parity(polynomial_q, k - 1), f"The parity of polynomial q is not satisfied.{polynomial_q.coef}"
    # check Theorem 3.3
    assert bool_function, f"The polynomial p and polynomial q are not normalized with max deviation {deviation}."
    phase_angle = np.zeros([k + 1])
    while degree_p > 0:
        # e^(2i*phi_k) = poly_p_{l} / poly_q{l-1}
        phase_angle[k] = np.log(polynomial_p.coef[degree_p] / polynomial_q.coef[degree_q]) / (2 * 1j)
        polynomial_p, polynomial_q = update_polynomial(phase_angle[k], polynomial_p, polynomial_q)
        k = k - 1
        degree_p = polynomial_p.degree()
        degree_q = polynomial_q.degree()

    # This is the trivial case, When degree p = 0, due to condition 3, we must have |p(1)| = 1
    # This also imply Q = 0.
    # Thus (phi_0, pi/2, -pi/2..., pi/2, -pi/2) is a solution.
    for i in range(1, k):
        phase_angle[i] = (-1)**(i + 1) * np.pi / 2
    phase_angle[0] = np.log(polynomial_p.coef[0]) / (1j)
    return phase_angle


# Following code convert angle sequence to another convention
# https://arxiv.org/abs/2002.11649
def convert_phase_sequence(phase_sequence: list[float]):
    """
    Convert the phase sequence in equation 13 to equation 15.

    Args:
        phase_sequence (list[float]): sequence of Rz rotation angle

    Returns:
        list[float]: new phase
    """
    new_phase = []
    for i in range(len(phase_sequence)):
        if i == 0 or i == len(phase_sequence) - 1:
            new_phase.append(phase_sequence[i] + np.pi / 4)
        else:
            new_phase.append(phase_sequence[i] + np.pi / 2)
    return new_phase


def negative_phase_sequence(phase_sequence: list[float]):
    """
    Convert the phase sequence in equation 13 to equation 18.
    Hence, generate -phi sequence used for U_(-phi)
    Args:
    phase_sequence (list[float]): sequence of Rz rotation angle

    Returns:
        list[float]: new phase
    """
    new_phase = []
    for i in range(len(phase_sequence)):
        if i == 0 or i == len(phase_sequence) - 1:
            new_phase.append(-1 * phase_sequence[i] + 3 * np.pi / 4)
        else:
            new_phase.append(-1 * phase_sequence[i] + np.pi / 2)
    return new_phase


def projector_controller(ancilla_size: int, angle: float):
    """
    make circuit in Figure 1, (b) of paper https://arxiv.org/abs/1806.01838.

    Args:
        ancilla_size (int): ancilla qubit size
        angle (float): the rotation angle embeds in Rz

    Returns:
        CompositeGate: composite gate calculate projector controller.
    """
    cg = CompositeGate()
    cg_x = CompositeGate()
    for i in range(ancilla_size):
        X | cg_x(i)
    ini = MultiControlToffoli("no_aux")
    CNX = ini(ancilla_size)
    # make cg gates to do e^(-iphi |0><0|^tensor n) where n is the ancilla size.
    cg_x | cg([i + 1 for i in range(ancilla_size)])
    CNX | cg([ancilla_size - i for i in range(ancilla_size + 1)])
    Rz(2 * angle) | cg(0)
    CNX | cg([ancilla_size - i for i in range(ancilla_size + 1)])
    cg_x | cg([i + 1 for i in range(ancilla_size)])
    return cg
