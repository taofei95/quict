import numpy as np
from numpy.polynomial.polynomial import Polynomial
from QuICT.core.gate import Rz, Rx
from QuICT.core.gate import CompositeGate
#######################################################

"""
Reference: https://arxiv.org/abs/1806.01838

In this file, we achieve this motivation.
given an input x in the domain[-1,1], and polynomial p(x)
We find the SIGNAL MATRIX
[p(x), iQ(x) * sqrt(1-x^2)]
[i*Q*(x)*sqrt(1-x^2), P*(x)]

Define a sequence of angle phi = {phi_0, phi_1..... phi_k} include in R_{k+1}
This matrix is yield by computing sequence of matrix
e^(i*phi_{0}*pauli_z) *product{k}_{j=1} [ W(x) * e^i*phi_{j}_pauli_z]

That means if we know the sequence of angle, we can block encode x and polynomial p(x) in 

This file provide following functionalities.
Given x and P(x)
1. find out Q(x)
2. find out angle_sequence
3. Compute signal matrix and generate signal circuit

The symbol used in the code is consistent with the symbol that we introduced above.
"""


class SignalAngleFinder:
    def __init__(self, tolerance=1e-6):

        self.tolerance = tolerance

    def execute(self, polynomial_p: Polynomial, k: int):
        """
        Given polynomial p find sequence of phi.
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

    def __init__(self, angle_sequence):
        self.angle_sequence = angle_sequence

    def signal_matrix(self, input_x):
        WX = W_X_marix(input_x)
        signal_matrix = exp_pauli_z_matrix(self.angle_sequence[0])
        for i in range(1, len(self.angle_sequence)):
            signal_matrix = np.matmul(signal_matrix, np.matmul(
                WX, exp_pauli_z_matrix(-self.angle_sequence[i])))
        return signal_matrix

    def signal_circuit(self, input_x):
        signal_x = -2*np.arccos(input_x)
        signal_composite_gate = CompositeGate()
        WX = Rx(signal_x)
        signal_gates = Rz(-2*self.angle_sequence[0])
        signal_gates | signal_composite_gate(0)
        for i in range(1, len(self.angle_sequence)):
            WX | signal_composite_gate(0)
            signal_gates = Rz(-2*self.angle_sequence[i])
            signal_gates | signal_composite_gate(0)
        return signal_composite_gate


def check_poly_parity(poly_p: Polynomial, k: int, tolerance: float = 1e-6):
    """
`    Check Theorem 3 condition 2.
    :param poly_p: polynomial p
    :param k: Input order
    :param tolerance: treat residue as zero.
    :return: Bool
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

    :param input_signal: input x
    :return: [x, i*sqrt(1-x^2)]
             [isqrt(1-x^2), x]
    """
    assert (-1 <= input_signal <= 1), "input_signal must be in domain[-1,1]"
    return np.array([[input_signal, np.sqrt(1-input_signal**2)*1j],
                     [np.sqrt(1-input_signal**2)*1j, input_signal]])


def exp_pauli_z_matrix(angle_phi: float):
    """
    generate e^(i*phi_{i} * pauli_z)
    :param angle_phi: Phi in the set{[phi_0, phi_1, phi_2...phi_k}
    :return: [e^i*angle_phi, 0]
             [0, e^-i*angle_phi]
    """
    return np.array([[np.exp(angle_phi*1j), 0],
                     [0, np.exp(angle_phi*-1j)]])


def block_encoding_matrix(signal_x: float, polynomial_p: Polynomial, polynomial_q: Polynomial):
    """
    Generate matrix [p(x), iQ(x) * sqrt(1-x^2)]
                    [i*Q*(x)*sqrt(1-x^2), P*(x)]
    :param signal_x: As the convention
    :param polynomial_p: P(x)
    :param polynomial_q: Q(x)
    """
    polynomial_q_conjugate = Polynomial(np.conj(polynomial_q.coef))
    polynomial_p_conjugate = Polynomial(np.conj(polynomial_p.ceof))
    return np.array([
        [polynomial_p(signal_x),
         1j*polynomial_q(signal_x)*np.sqrt(1-signal_x**2)],

        [1j*polynomial_q_conjugate(signal_x)*np.sqrt(1-signal_x**2),
         polynomial_p_conjugate(signal_x)]])


def check_polynomial_normalization(p: Polynomial, q: Polynomial, steps: int = 10, tolerance: float = 0.0001):
    """
    Check theorem condition 3.
    :param p: P(x)
    :param q: Q(x)
    :param steps: num of steps in range [-1, 1].
    :param tolerance: Max deviation from 1.
    :return: True if Theorem 3 satisfied, false if not satisfied.
    """
    summed_poly = p*Polynomial(np.conj(p.coef)) + \
        Polynomial([1, 0, -1])*q*Polynomial(np.conj(q.coef))
    x = np.linspace(-1, 1, steps)
    print(summed_poly(x))
    print(np.abs(np.abs(summed_poly(x))-1))
    max_deviation = np.max(np.abs(np.abs(summed_poly(x))-1))
    if max_deviation > tolerance:
        return False,  max_deviation
    return True, max_deviation


def check_angle_phi_quality(angle_phi: float,
                            polynomial_p: Polynomial,
                            polynomial_q: Polynomial,
                            steps: int = 100,
                            tolerance: float = 0.001):
    """
    check angle phi quality while updating polynomial as illustrated by equation 4.
    """
    x = np.linspace(-1, 1, steps)
    for i in range(steps):
        new_matrix = np.matmul(block_encoding_matrix(
            x[i], polynomial_p, polynomial_q), W_X_marix(x[i]))
        new_matrix = np.matmul(new_matrix, exp_pauli_z_matrix(angle_phi))
        test_polynomial_p = new_matrix[0][0]

        if np.abs(test_polynomial_p-polynomial_p(x[i])) > tolerance:
            print(np.abs(test_polynomial_p-polynomial_p(x[i])))
            return False
    return True


def complete_check(phi_list: np.ndarray, poly_p: Polynomial, tolerance=1e-6):
    """
    Check if the p(x) generated by signal processing which encoded in top left
     hand side of signal matrix is same as the input P(x).

    :param phi_list: phi sequence
    :param poly_p: P(x)
    :param tolerance: Max deviation from p(x)
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

    largest_error = np.max(np.abs(target_poly_p-np.array(calculated_poly_p)))
    if largest_error > tolerance:
        return False,  largest_error
    elif largest_error < tolerance:
        return True, largest_error


#######################################################
# following codes help generate polynomial Q who is the conjugate of polynomial p.

# The algorithm comes from Theorem 4.
def generate_A_tilt(polynomial_p: Polynomial):
    """
    A = 1-P(x)*P*(x). A is even polynomial. Set y=x^2
    A_tilt = A(y) = A(x^2)
    :return: A(y)
    """
    polynomial_p_conjugate = Polynomial(np.conj(polynomial_p.coef))
    A = 1-polynomial_p*polynomial_p_conjugate
    A_coefficient = A.coef
    A_tilt = Polynomial(A_coefficient[0::2])

    A_tilt_leading_coefficient = A_tilt.coef[A_tilt.degree()]
    return A_tilt, A_tilt_leading_coefficient


def generate_A_tilt_root(A_tilt: Polynomial, tolerance: float = 1e-12):
    """

    :param A_tilt: A(y)
    :param tolerance: All value below tolerance is treated as non significant roots
    :return: P(x) roots. Noting that complex roots comes in pairs and according to the theorem 4,
            We only need one roots in the complex multisets.
    """
    # All real root have even multiplicity except for one.
    # To avoid degeneracy root, we remove zero as well

    # get rid of 0 and 1.
    temp_roots = np.array([root for root in A_tilt.roots() if not ((np.abs(np.real(root)-1) < tolerance
                                                                    and np.abs(np.imag(root) < tolerance))
                                                                   or np.abs(root) < tolerance)])
    for i in range(len(temp_roots)):
        for j in range(len(temp_roots)):
            if (np.abs(np.real(temp_roots[i])-np.real(temp_roots[j])) < tolerance) and i != j:
                temp_roots = np.delete(temp_roots, j)
                break

        if len(temp_roots) == i+1:
            break
    return temp_roots


def generate_poly_q(polynomial_p: Polynomial, k: int):
    """
    Algorithm generate polynomial Q.
    Basically do W(y) = K * product_{set of roots s} [y-s] = K* product_{s} [x^2-s]
    K is a constant.
    """
    assert polynomial_p.degree() <= k, "Poly P order must larger than poly Q order."
    assert check_poly_parity(
        polynomial_p, k), "Poly P parity doens't satisfy the condition"
    A_tilt, A_tilt_leading_coefficient = generate_A_tilt(polynomial_p)
    A_tilt_roots = generate_A_tilt_root(A_tilt)
    # A_Y = Polynomial([1,-1])
    # for roots in A_tilt_roots:
    #     A_Y = Polynomial([-1*np.conj(roots), 1])*A_Y*Polynomial([-1*roots,1])
    # print(np.sqrt(nomalization_constant))

    # algorithm find Q
    polynomial_q = Polynomial([np.sqrt(-1*A_tilt_leading_coefficient)])
    for roots in A_tilt_roots:
        polynomial_q = polynomial_q*Polynomial([-1*roots, 0, 1])
    if k % 2 == 0:
        polynomial_q = polynomial_q * Polynomial([0, 1])
    return polynomial_q
###########################################################################################


def update_polynomial(phi: float, polynomial_p: Polynomial, polynomial_q: Polynomial):
    """
    Do calculation in equation 7 and equation 8.
    :return: updated p and q.
    """
    # The input polynomial must have odd-even properties.
    new_poly_p = np.exp(-1j * phi) * Polynomial([0, 1]) * polynomial_p + np.exp(
        1j * phi) * Polynomial([1, 0, -1]) * polynomial_q
    new_poly_q = np.exp(
        1j * phi) * Polynomial([0, 1]) * polynomial_q - np.exp(-1j * phi) * polynomial_p

    if new_poly_p.degree() >= polynomial_p.degree():
        new_poly_p.coef = np.delete(np.delete(new_poly_p.coef, -1), -1)
    if new_poly_q.degree() >= polynomial_q.degree() > 0:
        new_poly_q.coef = np.delete(np.delete(new_poly_q.coef, -1), -1)
    return new_poly_p, new_poly_q


def generate_phase_angle(polynomial_p: Polynomial, polynomial_q: Polynomial, k: int):
    """
    calculate sequence of phi.
    e^(2*i*phi) = p_l/q_{l-1}
    p_l is the last non zero term of polynomial p and q_l-1 is second last non zero term of polynomial q.
    :return: phase angle.
    """
    degree_p = polynomial_p.degree()
    degree_q = polynomial_q.degree()
    bool_function, deviation = check_polynomial_normalization(
        polynomial_p, polynomial_q)
    assert degree_p <= k, "The degree of polynomial p must lower or equal to k."
    assert degree_q <= (
        k-1), "The degree of polynomial q must lower or equal to k-1."
    assert check_poly_parity(
        polynomial_p, k), "The parity of polynomial p is not satisfied."
    assert check_poly_parity(
        polynomial_q, k-1), f"The parity of polynomial q is not satisfied.{polynomial_q.coef}"
    assert bool_function, f"The polynomial p and polynomial q are not normalized with max deviation {deviation}."

    phase_angle = np.zeros([k+1])

    # following is non-trivial case.
    while degree_p > 0:
        # e^(2i*phi_k) = poly_p_{l} / poly_q{l-1}
        phase_angle[k] = np.log(
            polynomial_p.coef[degree_p]/polynomial_q.coef[degree_q]) / (2*1j)
        if phase_angle[k] == 0:
            phase_angle[k] = np.pi

        polynomial_p, polynomial_q = update_polynomial(
            phase_angle[k], polynomial_p, polynomial_q)
        k = k-1
        degree_p = polynomial_p.degree()
        degree_q = polynomial_q.degree()
    # following is trivial case. If polynomial p degree is zero(constant function)
    # 1. phi_{0} = log(p)/i
    # 2. The order k must be even

    for i in range(1, k+1):
        phase_angle[i] = (-1)**(i+1)*np.pi/2
    phase_angle[0] = np.log(polynomial_p.coef[0])/(1j)
    return phase_angle
