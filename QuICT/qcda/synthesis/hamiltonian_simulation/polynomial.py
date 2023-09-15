import numpy as np
import scipy
from numpy.polynomial.polynomial import Polynomial
from numpy.polynomial.chebyshev import Chebyshev

def cheby_poly(order: int, input_value: float):
    """
    generate chebyshev polynomial
    Args:
        order(int): degrees of polynomial
        input_value(float): The x of polynomial(x)

    Returns:
        np.array: the value of chebyshev polynomial with given input value
                    Another word, Polynomial(x) value

    """
    poly_array = []
    for i in order:
        poly_array.append(sp.chebyt(i)(input_value))
    return np.array(poly_array)


def bessel_poly(order, input_value):
    """
    generate bessel polynomial
    Args:
        order(int): highest order of bessel polynomial
        input_value(float): x of f(x)

    Returns:
        np.array: bessel polynomial(x) value
    """
    return scipy.special.jv(order, input_value)


def exp_to_poly(time, order):
    """
    Convert e(i time ) to polynomials
    Args:
        time(float): t of e^(itH)
        order(int): degree of polynomial

    Returns:
        np.array: Polynomial coefficient from order 0 to largest order
    """
    coefficient_array = []
    coefficient_array.append(bessel_poly(0, time))
    for i in range(1, order):
        if i % 2 == 0:
            coefficient_array.append(2*(-1)**(i/2)*bessel_poly(i, time))
        elif i % 2 == 1:
            coefficient_array.append(2j*(-1)**((i-1)/2)*bessel_poly(i, time))

    return coefficient_array


class Poly:
    def __init__(self, polynomial="exp"):
        self.polynomial = polynomial

    def normal_basis(self, time: float, order: int):
        """
        Given input time and highest order. generate taylor expansion of e(i*time).
        Args:
            time(float): input parameter
            order(int): the highest order of chebyshev basis

        Returns:
            Polynomial: The chebyshev basis convert to normal basis

        """
        if self.polynomial == "exp":
            return Polynomial(np.polynomial.chebyshev.cheb2poly(exp_to_poly(time, order)))

    def chevbyshev_basis(self, time, order):
        """
        Given input time and highest order. generate taylor expansion of e(i*time).
        Args:
            time(float): input parameter
            order(int): the highest order of chebyshev basis

        Returns:
            Polynomial: The chebyshev basis polynomial

        """
        if self.polynomial == "exp":
            return Chebyshev(exp_to_poly(time, order))
