import numpy as np

from QuICT.core.gate import CompositeGate, CX, Rz, U1, GPhase
from QuICT.tools import Logger


_logger = Logger("DiagonalGate")


class DiagonalGate(object):
    """
    Diagonal gate

    Reference:
        https://arxiv.org/abs/2108.06150
    """
    _logger = _logger

    def __init__(self, target: int, aux: int = 0):
        """
        Args:
            target(int): number of target qubits
            aux(int, optional): number of auxiliary qubits
        """
        self.target = target
        if np.mod(aux, 2) != 0:
            self._logger.warn('Algorithm serves for even number of auxiliary qubits. One auxiliary qubit is dropped.')
            aux = aux - 1
        self.aux = aux

    def __call__(self, theta):
        """
        Args:
            theta(listLike): list of (2 ** target) angles of rotation in the diagonal gate

        Returns:
            CompositeGate: diagonal gate
        """
        assert len(theta) == 1 << self.target, ValueError('Incorrect number of angles')
        if self.aux == 0:
            return self.no_aux_qubit(theta)
        else:
            return self.with_aux_qubit(theta)

    def no_aux_qubit(self, theta):
        """
        Args:
            theta(listLike): list of (2 ** target) angles of rotation in the diagonal gate

        Returns:
            CompositeGate: diagonal gate without auxiliary qubit
        """

    def with_aux_qubit(self, theta):
        """
        Args:
            theta(listLike): list of (2 ** target) angles of rotation in the diagonal gate

        Returns:
            CompositeGate: diagonal gate with auxiliary qubit at the end of qubits
        """
        #Pay attention:
        #All arrays and qubit is 0 as the starting point, but begins with 1 in the paper.

        n = self.target
        m = self.aux
        gates = CompositeGate()

        for x in range(2**n):
        # Stage 1: Prefix Copy
            t = int(np.floor(np.log2(m / 2)))
            copies = int(np.floor(m / (2 * t)))
            r = int(np.floor(np.log2( copies + 1 )))
            rs = r + 1

            for i in range(1, rs):
                for j in range(t):
                    CX & [j, n + (2 ** (i - 1) - 1) * t + j] | gates
                for j in range((2 ** (i - 1) - 1) * t):
                    CX & [n + j, n + j + (2 ** (i - 1)) * t] | gates

            if 2 ** r - 1 < copies:
                rest = copies - 2 ** r + 1
                for j in range(t):
                    CX & [j, n + (2 ** (r - 1) - 1) * t + j] | gates
                if rest != 1:
                    for j in range((rest - 1) * t):
                        CX & [n + j, n + (2 ** (r - 1)) * t + j] | gates

            #for j in range(copies):
                #for i in range(t):
                    #CX & [i, n + i + j * t] | gates


         # Stage 2: Gray Initial
        #t = int(np.floor(np.log2(m / 2)))
            ell = 2**t
            ini_star = n + t * copies

            #1.implement U1
            for j in range(1, 1 + ell):
                for i in range(ini_star,ini_star+ell):
                    self.ket_fjk(j,1,n,t,i) | gates

             #2.implement R1
            s = self.partitioned_gray_code(n, t)
            for j in range(1,1+ell):
                sj1_int = int(s[j - 1][0], 2)
                phase= (self.linear_fjk(j,1,x,n,t)) * (self.alpha_s(theta,sj1_int,n))
                #phase = self.alpha_s(theta,sj1_int,n)
            #U1(phase) & (ini_star+j-1) | gates
                if phase != 0:
                    U1(phase) | gates(ini_star+j-1)
            #gates.extend(self.phase_shift_s(sj1_int, n, phase, aux=self.aux))

        #Stage 3:Suffix Copy

            #1.U^{\dagger}_{copy,1}
            for j in range(copies):
                for i in range(t):
                    CX & [i, n + i + j * t] | gates

            #2.U_{copy,2}
            copies3 = int(np.floor(m / (2 * (n-t))))

            for j in range(copies3):
                for i in range(n-t):
                    CX & [i + t, n + i + j * t] | gates

        #Stage 4: Gray Path
            num_phases = int((2** n)/ell)
            #the end label of the Stage 3
            sucoend = n + n - t - 1 + (copies3 - 1) * t
            for k in range(2,num_phases+1):

                #Step k.1: U_k
                for j in range(1,ell+1):
                    s = self.partitioned_gray_code(n,t)
                    s1 = s[j-1][k-2]
                    s2 = s[j-1][k-1]
                    for i in range(len(s1)):
                        if s1[i] != s2[i]:
                            CX & [i,sucoend+j] | gates
                            break

                   #Step k.2: R_k
                for j in range(1,ell+1):
                    s = self.partitioned_gray_code(n, t)
                    sjk_int = int(s[j-1][k-1],2)
                    #phase_k = self.alpha_s(theta,sjk_int,n)
                    phase_k = (self.linear_fjk(j,k,x,n,t)) * (self.alpha_s(theta,sjk_int,n))
                    if phase_k != 0:
                        U1(phase_k) | gates(j+sucoend)

                    #gates.extend(self.phase_shift_s(sjk_int, n, phase_k, aux=self.aux))

            #Stage 5:Inverse

            for j in range(copies3):
                for i in range(n-t):
                    CX & [i + t, n + i + j * t] | gates

            for j in range(1, ell + 1):
                s = self.partitioned_gray_code(n, t)
                s1 = s[j - 1][num_phases - 2]
                s2 = s[j - 1][num_phases - 1]
                for i in range(len(s1)):
                    if s1[i] != s2[i]:
                        y = n + n - t - 1 + (copies3 - 1) * t
                        CX & [i, y + j] | gates
                        break

        return gates

    @staticmethod
    def lucal_gray_code(k, n):
        """
        Generate the (k, n)-Gray code defined in and following Lemma 7

        Args:
            k(int): start the circular modification from the k-th binary code
            n(int): the length of binary code, that is, the length of Gray code would be 2^n

        Returns:
            iterable: the (k, n)-Gray code
        """
        def flip(bit):
            if bit == '1':
                return '0'
            if bit == '0':
                return '1'
            raise ValueError('Invalid bit found in gray code generation.')

        def zeta(x):
            """
            For integer x, zeta(x) = max{k: 2^k | x}
            """
            x_bin = np.binary_repr(x)
            return len(x_bin) - len(x_bin.strip('0'))

        gray_code = ['0' for _ in range(n)]
        yield ''.join(gray_code)
        for i in range(1, 1 << n):
            bit = np.mod(zeta(i) + k, n)
            gray_code[bit] = flip(gray_code[bit])
            yield ''.join(gray_code)

    @classmethod
    def partitioned_gray_code(cls, n, t):
        """
        Lemma 15 by the construction in Appendix E

        Args:
            n(int): length of 0-1 string to be partitioned
            t(int): length of the shared prefix of each row

        Returns:
            list[list[str]]: partitioned gray code
        """
        s = [[] for _ in range(1 << t)]
        for j in range(1 << t):
            prefix = np.binary_repr(j, width=t)[::-1]
            for suffix in cls.lucal_gray_code(np.mod(j, n - t), n - t):
                s[j].append(prefix + suffix)
        return s

    @staticmethod
    def binary_inner_prod(s, x, width):
        """
        Calculate the binary inner product of s_bin and x_bin,
        where s_bin and x_bin are binary representation of s and x respectively of width n

        Args:
            s(int): s in <s, x>
            x(int): x in <s, x>
            width(int): the width of s_bin and x_bin

        Returns:
            int: the binary inner product of s and x
        """
        s_bin = np.array(list(np.binary_repr(s, width=width)), dtype=int)
        x_bin = np.array(list(np.binary_repr(x, width=width)), dtype=int)
        return np.mod(np.dot(s_bin, x_bin), 2)

    @classmethod
    def alpha_s(cls, theta, s, n):
        """
        Solve Equation 6
        \sum_s alpha_s <s, x> = theta(x)

        Args:
            theta(listLike): phase angles of the diagonal gate
            s(int): key of the solution component
            n(int): number of qubits in the diagonal gate

        Returns:
            float: alpha_s in Equation 6
        """
        A = np.zeros(1 << n)
        for x in range(1, 1 << n):
            A[x] = cls.binary_inner_prod(s, x, width=n)
        # A_inv = 2^(1-n) (2A - J)
        #A_inv = (2 * A[1:] - 1) / (1 << (n - 1))
        # As size should be matched,we change the code
        A_inv = (2 * A[0:] - 1) / (1 << (n - 1))
        return np.dot(A_inv, theta)

    @classmethod
    def phase_shift(cls, theta, seq=None, aux=None):
        """
        Implement the phase shift
        |x> -> exp(i theta(x)) |x>
        by solving Equation 6
        \sum_s alpha_s <s, x> = theta(x)

        Args:
            theta(listLike): phase angles of the diagonal gate
            seq(iterable, optional): sequence of s application, numerical order if not assigned
            aux(int, optional): key of auxiliary qubit (if exists)

        Returns:
            CompositeGate: CompositeGate of the diagonal gate
        """
        n = int(np.floor(np.log2(len(theta))))
        if seq is None:
            seq = range(1, 1 << n)
        else:
            assert sorted(list(seq)) == list(range(1, 1 << n)), ValueError('Invalid sequence of s in phase_shift')
        if aux is not None:
            assert aux >= n, ValueError('Invalid auxiliary qubit in phase_shift.')
        # theta(0) = 0
        global_phase = theta[0]
        theta = (theta - global_phase)[1:]

        gates = CompositeGate()
        GPhase(global_phase) & 0 | gates
        # Calculate A_inv row by row (i.e., for different s)
        for s in seq:
            alpha = cls.alpha_s(theta, s, n)
            if aux is not None:
                gates.extend(cls.phase_shift_s(s, n, alpha, aux=aux))
            else:
                gates.extend(cls.phase_shift_s(s, n, alpha, j=0))
        return gates

    @classmethod
    def phase_shift_s(cls, s, n, alpha, aux=None, j=None):
        """
        Implement the phase shift for a certain s defined in Equation 5 as Figure 8
        |x> -> exp(i alpha_s <s, x>) |x>

        Args:
            s(int): whose binary representation stands for the 0-1 string s
            n(int): the number of qubits in |x>
            alpha(float): alpha_s in the equation
            aux(int, optional): key of auxiliary qubit (if exists)
            j(int, optional): if no auxiliary qubit, the j-th smallest element in s_idx would be the target qubit

        Returns:
            CompositeGate: CompositeGate for Equation 5 as Figure 8
        """
        gates = CompositeGate()
        s_bin = np.binary_repr(s, width=n)
        s_idx = []
        for i in range(n):
            if s_bin[i] == '1':
                s_idx.append(i)

        # Figure 8 (a)
        if aux is not None:
            if j is not None:
                cls._logger.warn('With auxiliary qubit in phase_shift_s, no i_j is needed.')
            assert aux >= n, ValueError('Invalid auxiliary qubit in phase_shift_s.')
            for i in s_idx:
                CX & [i, aux] | gates
            U1(alpha) & aux | gates
            for i in reversed(s_idx):
                CX & [i, aux] | gates
            return gates
        # Figure 8 (b)
        else:
            assert j < len(s_idx), ValueError('Invalid target in phase_shift without auxiliary qubit.')
            for i in s_idx:
                if i == s_idx[j]:
                    continue
                CX & [i, s_idx[j]] | gates
            U1(alpha) & s_idx[j] | gates
            for i in s_idx:
                if i == s_idx[j]:
                    continue
                CX & [i, s_idx[j]] | gates
            return gates

    @classmethod
    def linear_fjk(cls,j,k,x,n,t):
        """
        implement the linear functions f_{jk}(x) =<s(j,k),x>

        Args:
            j(int):j is the label of n-bit strings s(j,k)
            k(int):k is the label of n-bit strings s(j,k)
            n(int): length of 0-1 string to be partitioned
            t(int): length of the shared prefix of each row
            x(int):the independent variables of the function f_{jk}
        """
        s = cls.partitioned_gray_code(n, t)
        decimal_integer = int(s[j - 1][k - 1], 2)  # Convert the binary string s[j - 1][k-1] to an integer
        return cls.binary_inner_prod(decimal_integer, x, width=n)

    @classmethod
    def ket_fjk(cls,j,k,n,t,target_num):
        """
        implement the part of unitary U1 for every j:
        |0> -> |<s(j,k),x>> by adding the CNOT gates

        Args:
            j(int):j is the label of n-bit strings s(j,k)
            k(int):k is the label of n-bit strings s(j,k)
            n(int): length of 0-1 string to be partitioned
            t(int): length of the shared prefix of each row
            target_num(int):the target label connecting the CNOT gate

        return:
            CompositeGate
        """

        s=cls.partitioned_gray_code(n, t)
        st=s[j-1][k-1]
        st_idx = []

        gates = CompositeGate()

        for i in range(len(st)):
            if st[i] == '1':
                #st_idx.append(i)
                CX & [i, target_num] | gates

        return gates
