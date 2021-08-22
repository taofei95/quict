"""
A Qubit operator is a polynomial of Pauli matrices {X, Y, Z} = {sigma_1, sigma_2, sigma_3}, 
which is a useful representation for circuits by second quantization. 
"""

from polynomial_operator import PolynomialOperator

class QubitOperator(PolynomialOperator):
    """    
    A Qubit operator is a polynomial of Pauli matrices {X, Y, Z} = {sigma_1, sigma_2, sigma_3}, 
    which is a useful representation for circuits by second quantization. 

    In this class, the operator could be represented as below.
    For example, list
    [[[(i, 1), (j, 1), (k, 3), (l, 2)], 1.2], [[(i, 1), (j, 3), (s,2)], -1.2], ...]
    stands for '1.2 Xi Xj Zk Yl - 1.2 Xi Zj Ys + ...',

    In the following descriptions, the above list is called list format,
    while the above string is called string format.
    """
    def __init__(self, monomial=None, coefficient=1.):
        """
        Create a monomial of ladder operators with the two given formats.

        Args:
            monomial(list/str): Operator monomial in list/string format
            coefficient(int/float/complex): Coefficient of the monomial
        """
        super().__init__(monomial,coefficient)
        if self.operators == []:
            return
        variables = self.operators[0][0]
        l = len(variables)

        # The second parameter(kind) in fermion operator should be {1,2,3}
        if any([var[1] not in [1,2,3] for var in variables]):
            raise Exception("Illegal qubit operator.")

        # The variables in a monomial should be in ascending order.
        # Commutation relation for operators on different targets
        for i in range(l-1, 0, -1):
            fl = False
            for j in range(i):
                if variables[j][0] > variables[j+1][0]:
                    variables[j], variables[j+1] = variables[j+1], variables[j]
                    fl = True
            if not fl:
                break

        # Commutation relation for operators on identical targets
        operators = []
        for i in range(l):
            if i == 0 or variables[i][0] != variables[i-1][0]:
                cur = variables[i][1]
                j = i + 1
                while j < l and variables[j][0] == variables[i][0]:
                    if cur == 0:
                        cur = variables[j][1]
                    elif cur == variables[j][1]:
                        cur = 0
                    else:
                        coefficient *= complex(0, (-1)**((cur - variables[j][1] + 3) % 3))
                        cur = 6 - cur - variables[j][1]
                    j += 1
                if cur != 0:
                    operators += [(variables[i][0], cur)]
        self.operators = [[operators, coefficient]]

    @classmethod
    def getPolynomial(cls, monomial=None, coefficient=1.):
        '''
        Construct an instance of the same class as 'self'

        Args:
            monomial(list/str): Operator monomial in list/string format
            coefficient(int/float/complex): Coefficient of the monomial
        '''
        return QubitOperator(monomial, coefficient)

    @classmethod
    def analyze_single(cls, single_operator):
        """
        Transform a string format of a single operator to a tuple

        Args:
            single_operator(str): string format

        Returns:
            tuple: the corresponding tuple in list format
        """
        if single_operator[0] == 'X':
            return (int(single_operator[1:]), 1)
        elif single_operator[0] == 'Y':
            return (int(single_operator[1:]), 2)
        elif single_operator[0] == 'Z':
            return (int(single_operator[1:]), 3)
        else:
            raise Exception("The string format is not recognized: "+single_operator)

    @classmethod
    def parse_single(cls, single_operator):
        """
        Transform a tuple format of a single operator to a string

        Args:
            single_operator(tuple): list format

        Returns:
            string: the corresponding string format
        """
        if single_operator[1]==1:
            return 'X'+str(single_operator[0])+' '
        elif single_operator[1]==2:
            return 'Y'+str(single_operator[0])+' '
        elif single_operator[1]==3:
            return 'Z'+str(single_operator[0])+' '
        
