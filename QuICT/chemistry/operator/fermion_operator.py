"""
A fermion operator is a polynomial of anti-commutative creation-annihilation operators, 
which is a useful representation for states and Hamiltonians by second quantization. 
"""

import numpy as np

class FermionOperator(object):
    """    
    A fermion operator is a polynomial of anti-commutative creation-annihilation operators, 
    which is a useful representation for states and Hamiltonians by second quantization. 

    Due to the anti-commutation relation, the polynomial is in fact a multilinear function of
    the ladder operators. In this class, the operator could be represented as below.

    For example, tuple
    ((((i, 1), (j, 1), (k, 0), (l, 0)), 1.2), (((k, 1), (l, 1), (i, 0), (j, 0)), -1.2), ...)
    stands for (1.2 a_i^\dagger a_j^\dagger a_k a_l - 1.2 a_k^\dagger a_l^\dagger a_i a_j + ...),
    which could also be parsed as string '1.2 * i^ j^ k l - 1.2 * k^ l^ i j + ...'.

    In the following descriptions, the above tuple is called tuple format, while the above string
    is called string format.
    """
    def __init__(self, monomial=None, coefficient=1.):
        """
        Create a monomial of ladder operators with the two given formats.

        Args:
            monomial(tuple/str): Operator monomial in tuple/string format
            coefficient(int/float/complex): Coefficient of the monomial
        """
        self.operators = ()

    def __add__(self, other):
        """
        Addition of two operators

        Args:
            other(FermionOperator): Operator to be added

        Returns:
            FermionOperator: self + other
        """
        pass

    def __iadd__(self, other):
        """
        Implement the '+=' operation
        """
        pass

    def __radd__(self, other):
        """
        Args:
            other(FermionOperator): Operator to be added

        Returns:
            FermionOperator: other + self
        """
        pass

    def __sub__(self, other):
        """
        Substraction of two operators

        Args:
            other(FermionOperator): Operator to be substracted

        Returns:
            FermionOperator: self - other
        """
        pass

    def __isub__(self, other):
        """
        Implement the '-=' operation
        """
        pass

    def __rsub__(self, other):
        """
        Args:
            other(FermionOperator): Operator to be substracted

        Returns:
            FermionOperator: other - self
        """
        pass

    def __mul__(self, other):
        """
        Multiplication of two operators or an operator with a number

        Args:
            other(FermionOperator/int/float/complex): multiplier

        Returns:
            FermionOperator: self * other
        """
        pass

    def __imul__(self, other):
        """
        Implement the '*=' operation
        """
        pass

    def __rmul__(self, other):
        """
        Args:
            other(FermionOperator/int/float/complex): multiplier

        Returns:
            FermionOperator: other * self
        """
        pass

    def __truediv__(self, other):
        """
        Division of an operator with a number

        Args:
            other(int/float/complex): divisor

        Returns:
            FermionOperator: self / other
        """
        pass

    def __itruediv__(self, other):
        """
        Implement the '/=' operation
        """
        pass
