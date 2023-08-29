import numpy as np
from numpy.polynomial.polynomial import Polynomial as P
from numpy.polynomial.chebyshev import Chebyshev as C
import scipy.special as sp


def cheby_poly(order, input_value):
    """
    generate chebyshev polynomial

    Parameters
    ----------
    order : list or 1-D array
        degrees of polynomial
    input_value : float
        The x of polynomial(x)

    Returns
    -------
    Array
        the value of chebyshev polynomial with given input value
        Another word, Polynomial(x) value

    """
    poly_array = []
    for i in order:
        poly_array.append(sp.chebyt(i)(input_value))
    return np.array(poly_array)


def bessel_poly(order, input_value):
    """
    generate bessel polynomial

    Parameters
    ----------
    order : highest order of bessel polynomial

    input_value : FLoat
        x of f(x)
    Returns
    -------
    bessel polynomial(x) value

    """
    return sp.jv(order, input_value)


def exp_to_poly(time, order):
    """
    Convert e(i time ) to polynomials

    Parameters
    ----------
    time : float
    order : int
            degree of polynomial

    Returns
    -------
    coefficient_array : 1-D array
        Polynomial coefficient from order 0 to largest order

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

    def normal_basis(self, time, order: int):
        """
        Given input time and max order. generate taylor expansion of e(i*t).

        """
        if self.polynomial == "exp":
            return P(np.polynomial.chebyshev.
                     cheb2poly(exp_to_poly(time, order)))

    def chevbyshev_basis(self, time, order):
        """
        Given input time
        generate polynomial expansion of e(i*time) in chevbyshev  kind basis

        """
        if self.polynomial == "exp":
            return C(exp_to_poly(time, order))
