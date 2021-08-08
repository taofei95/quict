"""
A fermion operator is a polynomial of anti-commutative creation-annihilation operators, 
which is a useful representation for states and Hamiltonians by second quantization. 
"""

import numpy as np
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
        vars = []
        if isinstance(monomial, list):
            vars = copy.deepcopy(monomial)         #########
        elif isinstance(monomial, str):
            for var in monomial.split():
                if var[-1] == '^':
                    vars.append((int(var[:-1]), 1))
                else:
                    vars.append((int(var), 0))

        # The variables in a monomial should be in ascending order. So do monomials in the polynomial.
        l = len(vars)
        for i in range(l-1, 0, -1):
            fl = False
            for j in range(i):
                if vars[j][0] > vars[j+1][0]:
                    vars[j], vars[j+1] = vars[j+1], vars[j]
                    coefficient *= -1
                    fl = True
            if not fl:
                break

        # Adjacent identical operators lead to zero operator
        self.operators = []
        if any([vars[i] == vars[i+1] for i in range(l - 1)]):
            return

        # Anti-commutation relation
        for i in range(l):
            if i == 0 or vars[i][0] != vars[i-1][0]:
                k = l
                for j in range(i+1, l):
                    if vars[j][0] != vars[i][0]:
                        k = j
                        break
                if (k - i) % 2 == 1:
                    self.operators += [vars[i]]
                elif vars[i][1] == 1:
                    self.operators += [vars[i], vars[i+1]]
                else:
                    #self = (FermionOperator(self.operators, coefficient) + FermionOperator(self.operators + [vars[i+1], vars[i]], -coefficient)) * FermionOperator(vars[k:])
                    self.operators = [[self.operators, coefficient], [self.operators + [vars[i+1], vars[i]], -coefficient]]
                    #print(self.operators)
                    self *= FermionOperator(vars[k:]) # mul promises the ascending order
                    #print(self.operators)
                    print(self)
                    return # polynomial
        self.operators = [[self.operators, coefficient]]
        print(self)
        # monomial

    def __add__(self, other):
        """
        Addition of two operators
        Args:
            other(FermionOperator): Operator to be added
        Returns:
            FermionOperator: self + other
        """
        ans=FermionOperator(0)
        A = self.operators
        B = other.operators
        ia = ib = 0
        la = len(A)
        lb = len(B)
        while ia < la and ib < lb:
            if A[ia][0] == B[ib][0]:
                if A[ia][1]+B[ib][1] != 0:
                    ans.operators += [[copy.deepcopy(A[ia][0]),A[ia][1]+B[ib][1]]]
                ia += 1
                ib += 1
            elif A[ia][0]<B[ib][0]:
                ans.operators += [copy.deepcopy(A[ia])]
                ia += 1
            else:
                ans.operators += [copy.deepcopy(B[ib])]
                ib += 1
        while ia < la:
            ans.operators += [copy.deepcopy(A[ia])]
            ia += 1
        while ib < lb:
            ans.operators += [copy.deepcopy(B[ib])]
            ib += 1
        return ans

    def __iadd__(self, other):
        """
        Implement the '+=' operation
        """
        self = self + other
        return self

    def __radd__(self, other):
        """
        Args:
            other(FermionOperator): Operator to be added
        Returns:
            FermionOperator: other + self
        """
        return self + other

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
        self = self - other
        return self

    def __rsub__(self, other):
        """
        Args:
            other(FermionOperator): Operator to be substracted
        Returns:
            FermionOperator: other - self
        """
        return other + self * (-1)

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
            ans.operators += [[copy.deepcopy(mono[0]), mono[1] * other] for mono in self.operators]
            return ans
        A = self.operators
        B = other.operators
        for ma in A:
            for mb in B:
                '''
                va = ma[0]
                vb = mb[0]
                vc = []
                coe = ma[1] * mb[1]
                ia = ib = 0
                la = len(va)
                lb = len(vb)
                while ia < la and ib < lb:
                    if (va[ia][0] == vb[ib][0]):
                        vc += [va[ia], vb[ib]]
                        ia += 1
                        ib += 1
                        coe *= (-1) ** (la - ia)
                    elif (va[ia][0] < vb[ib][0]):
                        vc += [va[ia]]
                        ia += 1
                    else:
                        vc += [vb[ib]]
                        ib += 1
                        coe *= (-1) ** (la - ia)
                while ia < la:
                    vc += [va[ia]]
                    ia += 1
                while ib < lb:
                    vc += [vb[ib]]
                    ib += 1
                '''
                ans += FermionOperator(ma[0] + mb[0], ma[1] * mb[1])
        return ans
        

    def __imul__(self, other):
        """
        Implement the '*=' operation
        """
        self = self * other
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
        self = self / other
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

fa = FermionOperator("1^ 4^ 13 2 405 2^ ",-1.2)
print(fa)
print(fa.parse())
#fb=FermionOperator("  1^ 2^ ",2)
#print(fb)
#print((f+fb).operators)