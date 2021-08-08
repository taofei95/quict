"""
A fermion operator is a polynomial of anti-commutative creation-annihilation operators, 
which is a useful representation for states and Hamiltonians by second quantization. 
"""

import copy

class FermionOperator(object):
    """    
    A fermion operator is a polynomial of anti-commutative creation-annihilation operators, 
    which is a useful representation for states and Hamiltonians by second quantization. 

    Due to the anti-commutation relation, the polynomial is in fact a multilinear function of
    the ladder operators. In this class, the operator could be represented as below.
    For example, list
    [[[(i, 1), (j, 1), (k, 0), (l, 0)], 1.2], [[(k, 1), (l, 1), (i, 0), (j, 0)], -1.2], ...]
    stands for (1.2 a_i^\dagger a_j^\dagger a_k a_l - 1.2 a_k^\dagger a_l^\dagger a_i a_j + ...),
    which could also be parsed as string '1.2 * i^ j^ k l - 1.2 * k^ l^ i j + ...'.

    In the following descriptions, the above list is called list format, while the above string
    is called string format.
    """
    def __init__(self, monomial=None, coefficient=1.):
        """
        Create a monomial of ladder operators with the two given formats.

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
                if var[-1] == '^':
                    variables.append((int(var[:-1]), 1))
                else:
                    variables.append((int(var), 0))

        # The variables in a monomial should be in ascending order.
        # So do monomials in the polynomial.
        l = len(variables)
        for i in range(l-1, 0, -1):
            fl = False
            for j in range(i):
                if variables[j][0] > variables[j+1][0]:
                    variables[j], variables[j+1] = variables[j+1], variables[j]
                    coefficient *= -1
                    fl = True
            if not fl:
                break

        # Adjacent identical operators lead to zero operator
        self.operators = []
        if any([variables[i] == variables[i+1] for i in range(l - 1)]):
            return

        # Anti-commutation relation
        for i in range(l):
            if i == 0 or variables[i][0] != variables[i-1][0]:
                k = l
                for j in range(i+1, l):
                    if variables[j][0] != variables[i][0]:
                        k = j
                        break
                if (k - i) % 2 == 1:
                    self.operators += [variables[i]]
                elif variables[i][1] == 1:
                    self.operators += [variables[i], variables[i+1]]
                else:
                    self.operators = [[self.operators, coefficient], [self.operators + [variables[i+1], variables[i]], -coefficient]]
                    self *= FermionOperator(variables[k:]) # mul promises the ascending order
                    return
        self.operators = [[self.operators, coefficient]]

    def __add__(self, other):
        """
        Addition of two operators

        Args:
            other(FermionOperator): Operator to be added

        Returns:
            FermionOperator: self + other
        """
        ans = FermionOperator(0)
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
        A = copy.deepcopy(self.operators)
        B = other.operators
        self.operators = []
        ia = ib = 0
        while ia < len(A) and ib < len(B):
            if A[ia][0] == B[ib][0]:
                if A[ia][1]+B[ib][1] != 0:
                    self.operators += [[copy.deepcopy(A[ia][0]),A[ia][1]+B[ib][1]]]
                ia += 1
                ib += 1
            elif A[ia][0]<B[ib][0]:
                self.operators += [copy.deepcopy(A[ia])]
                ia += 1
            else:
                self.operators += [copy.deepcopy(B[ib])]
                ib += 1
        while ia < len(A):
            self.operators += [copy.deepcopy(A[ia])]
            ia += 1
        while ib < len(B):
            self.operators += [copy.deepcopy(B[ib])]
            ib += 1
        return self

    def __sub__(self, other):
        """
        Substraction of two operators

        Args:
            other(FermionOperator): Operator to be substracted

        Returns:
            FermionOperator: self - other
        """
        return self + other * (-1)

    def __isub__(self, other):
        """
        Implement the '-=' operation
        """
        self += other * (-1)
        return self

    def __mul__(self, other):
        """
        Multiplication of two operators or an operator with a number

        Args:
            other(FermionOperator/int/float/complex): multiplier

        Returns:
            FermionOperator: self * other
        """
        ans = FermionOperator(0)
        if not isinstance(other, FermionOperator):
            ans.operators = [[copy.deepcopy(mono[0]), mono[1] * other] for mono in self.operators]
            return ans
        A = self.operators
        B = other.operators
        for mono_A in A:
            for mono_B in B:
                ans += FermionOperator(mono_A[0] + mono_B[0], mono_A[1] * mono_B[1])
        return ans

    def __imul__(self, other):
        """
        Implement the '*=' operation
        """
        if not isinstance(other, FermionOperator):
            for mono in self.operators:
                mono[1] *= other
            return self
        A = copy.deepcopy(self.operators)
        B = other.operators
        self.operators = []
        for mono_A in A:
            for mono_B in B:
                self += FermionOperator(mono_A[0] + mono_B[0], mono_A[1] * mono_B[1])
        return self

    def __rmul__(self, other):
        """
        Args:
            other(FermionOperator/int/float/complex): multiplier

        Returns:
            FermionOperator: other * self
        """
        if not isinstance(other, FermionOperator):
            return self * other
        return other * self

    def __truediv__(self, other):
        """
        Division of an operator with a number

        Args:
            other(int/float/complex): divisor

        Returns:
            FermionOperator: self / other
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
        ans=''
        for mono in self.operators:
            ans += '+ ('+str(mono[1])+') '
            if mono[0] != []:
                ans += '* '
                for var in mono[0]:
                    ans += str(var[0]) + ('^ ' if var[1] else ' ')
        return ans
