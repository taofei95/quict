import numpy as np
from numpy.polynomial.polynomial import Polynomial as P
import numpy.polynomial.chebyshev as C
import scipy.special as sp
def cheby_poly(order, input_value):
    poly_array = []
    for i in order:
        poly_array.append(sp.chebyt(i)(input_value))
    return np.array(poly_array)

def bessel_poly(order, input_value):
    return sp.jv(order, input_value)


def exp_to_poly(time, order):
    coefficient_array = []
    coefficient_array.append(bessel_poly(0, time))
    for i in range(1, order):
        if i%2==0:
            coefficient_array.append(2*(-1)**(i/2)*bessel_poly(i, time))
        elif i%2 ==1:
            coefficient_array.append(2j*(-1)**((i-1)/2)*bessel_poly(i,time))

    return coefficient_array

class Poly:
    def __init__(self, polynomial="exp"):
        self.polynomial = polynomial
    def normal_basis(self,time, order:int):
        if self.polynomial =="exp":
            return P(C.cheb2poly(exp_to_poly(time, order)))


    def chevbyshev_basis(self):
        if self.polynomial =="exp":
            return C(exp_to_poly(time, order))