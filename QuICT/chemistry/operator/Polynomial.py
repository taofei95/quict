"""
A polynomial of symbolic operators.
The superClass of FermionOperator and QubitOperator.
"""

import copy

class Polynomial(object):
    """    
    A polynomial of symbolic operators.
    The superClass of FermionOperator and QubitOperator.

    In this class, the operator could be represented as below.
    The first parameter in a binary tuple represents the target, while the second one represents the type.
    Two monomials can be merged if and only if their list formats only differ in the coefficient.
    The standardlization for a monomial should be accomplished in __init__ according to specific rules.
    
    For example, list
    [[[(i, 1), (j, 1), (k, 0), (l, 0)], 1.2], [[(k, 1), (l, 1), (i, 0), (j, 0)], -1.2], ...]
    stands for '1.2 a_i^(1) a_j^(1) a_k^(0) a_l^(0) - 1.2 a_k^(1) a_l^(1) a_i^(0) a_j^(0) + ...',

    In the following descriptions, the above list is called list format,
    while the above string is called string format.
    """
    def __init__(self, monomial=None, coefficient=1.):
        """
        Create a monomial of operators with the two given formats.

        Args:
            monomial(list/str): Operator monomial in list/string format
            coefficient(int/float/complex): Coefficient of the monomial
        """
        # If monomial is None, it means identity operator.(not the zero operator)
        # If monomial is 0, it means the zero operator
        if monomial == 0 or coefficient == 0:
            self.operators = []
            return
        if monomial == None:
            self.operators = [[[], coefficient]]
            return
        variables = []
        if isinstance(monomial, list):
            variables = copy.deepcopy(monomial)
        elif isinstance(monomial, str):
            for var in monomial.split():
                variables.append(self.str2tuple(var))
        self.operators=[[variables,coefficient]]
    
    def getPolynomial(self, monomial=None, coefficient=1.):
        '''
        Construct an instance of the same class as 'self'

        Args:
            monomial(list/str): Operator monomial in list/string format
            coefficient(int/float/complex): Coefficient of the monomial
        '''
        raise Exception("Construction of Polynomial is prohibited")

    def str2tuple(cls, single_operator):
        """
        Transform a string format of a single operator to a tuple

        Args:
            single_operator(str): string format

        Returns:
            tuple: the corresponding tuple in list format
        """
        raise Exception("The string format is not recognized.")

    def tuple2str(cls, single_operator):
        """
        Transform a tuple format of a single operator to a string

        Args:
            single_operator(tuple): list format

        Returns:
            string: the corresponding string format
        """
        raise Exception("The string format is not realized.")

    def __add__(self, other):
        """
        Addition of two operators

        Args:
            other(Polynomial): Operator to be added

        Returns:
            Polynomial: self + other
        """
        ans = self.getPolynomial(0)
        A = self.operators
        B = other.operators
        ia = ib = 0
        lenA = len(A)
        lenB = len(B)
        while ia < lenA and ib < lenB:
            if A[ia][0] == B[ib][0]:
                if A[ia][1] + B[ib][1] != 0:
                    ans.operators += [[copy.deepcopy(A[ia][0]),A[ia][1]+B[ib][1]]]
                ia += 1
                ib += 1
            elif A[ia][0]<B[ib][0]:
                ans.operators += [copy.deepcopy(A[ia])]
                ia += 1
            else:
                ans.operators += [copy.deepcopy(B[ib])]
                ib += 1
        while ia < lenA:
            ans.operators += [copy.deepcopy(A[ia])]
            ia += 1
        while ib < lenB:
            ans.operators += [copy.deepcopy(B[ib])]
            ib += 1
        return ans

    def __iadd__(self, other):
        """
        Implement the '+=' operation
        """
        ans = self + other
        self.operators = ans.operators
        return self

    def __mul__(self, other):
        """
        Multiplication of two operators or an operator with a number

        Args:
            other(Polynomial/int/float/complex): multiplier

        Returns:
            Polynomial: self * other
        """
        ans = self.getPolynomial(0)
        if not isinstance(other, Polynomial):
            ans.operators = [[copy.deepcopy(mono[0]), mono[1] * other] for mono in self.operators]
            return ans
        A = self.operators
        B = other.operators
        for mono_A in A:
            for mono_B in B:
                ans += self.getPolynomial(mono_A[0] + mono_B[0], mono_A[1] * mono_B[1])
        return ans

    def __imul__(self, other):
        """
        Implement the '*=' operation
        """
        ans = self * other
        self.operators = ans.operators
        return self

    def __rmul__(self, other):
        """
        Args:
            other(Polynomial/int/float/complex): multiplier

        Returns:
            Polynomial: other * self
        """
        if not isinstance(other, Polynomial):
            return self * other
        return other * self
    
    def __sub__(self, other):
        """
        Substraction of two operators

        Args:
            other(Polynomial): Operator to be substracted

        Returns:
            Polynomial: self - other
        """
        return self + other * (-1)

    def __isub__(self, other):
        """
        Implement the '-=' operation
        """
        self += other * (-1)
        return self

    def __truediv__(self, other):
        """
        Division of an operator with a number

        Args:
            other(int/float/complex): divisor

        Returns:
            Polynomial: self / other
        """
        return self * (1./other)

    def __itruediv__(self, other):
        """
        Implement the '/=' operation
        """
        self *= (1./other)
        return self

    def __repr__(self) -> str:
        """
        Return the string format
        """
        return f"{self.operators}"

    def parse(self):
        """
        Parse the list format to string format
        """
        if self.operators == []:
            return '0'
        ans = ''
        for mono in self.operators:
            ans += '+ (' + str(mono[1]) + ') '
            if mono[0] != []:
                ans += '* '
                for var in mono[0]:
                    ans += self.tuple2str(var)
        return ans
