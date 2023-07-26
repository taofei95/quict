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

    def __call__(self, angles):
        """
        Args:
            angles(listLike): list of (2 ** target) angles of rotation in the diagonal gate

        Returns:
            CompositeGate: diagonal gate
        """
        assert len(angles) == 1 << self.target, ValueError('Incorrect number of angles')
        if self.aux == 0:
            return self.no_aux_qubit(angles)
        else:
            return self.with_aux_qubit(angles)

    def no_aux_qubit(self, angles):
        """
        Args:
            angles(listLike): list of (2 ** target) angles of rotation in the diagonal gate

        Returns:
            CompositeGate: diagonal gate without auxiliary qubit
        """

    def with_aux_qubit(self, angles):
        """
        Args:
            angles(listLike): list of (2 ** target) angles of rotation in the diagonal gate

        Returns:
            CompositeGate: diagonal gate with auxiliary qubit at the end of qubits
        """
        gates = CompositeGate()

        # Stage 1: Prefix Copy
        t = np.floor(np.log2(self.aux / 2))
        copies = np.floor(self.aux / (2 * t))
        for j in range(copies):
            for i in range(t):
                CX & [i, self.target + i * copies + j] | gates

        # Stage 2: Gray Initial

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
    def phase_shift(cls, theta, aux=None):
        """
        Implement the phase shift
        |x> -> exp(i theta(x)) |x>
        by solving Equation 6
        \sum_s alpha_s <s, x> = theta(x)

        Args:
            theta(listLike): phase angles of the diagonal gate
            aux(int, optional): key of auxiliary qubit (if exists)

        Returns:
            CompositeGate: CompositeGate of the diagonal gate
        """
        n = int(np.floor(np.log2(len(theta))))
        if aux is not None:
            assert aux >= n, ValueError('Invalid auxiliary qubit in phase_shift.')
        # theta(0) = 0
        global_phase = theta[0]
        theta = (theta - global_phase)[1:]

        gates = CompositeGate()
        GPhase(global_phase) & 0 | gates
        # Calculate A_inv row by row (i.e., for different s)
        for s in range(1, 1 << n):
            A = np.zeros(1 << n)
            for x in range(1, 1 << n):
                A[x] = cls.binary_inner_prod(s, x, width=n)
            # A_inv = 2^(1-n) (2A - J)
            A_inv = (2 * A[1:] - 1) / (1 << (n - 1))
            alpha_s = np.dot(A_inv, theta)
            if aux is not None:
                gates.extend(cls.phase_shift_s(s, n, alpha_s, aux=aux))
            else:
                gates.extend(cls.phase_shift_s(s, n, alpha_s, j=0))
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
            j(int, optional): if no auxiliary qubit, the j-th smallest element in S would be the target qubit

        Returns:
            CompositeGate: CompositeGate for Equation 5 as Figure 8
        """
        gates = CompositeGate()
        s_bin = np.binary_repr(s, width=n)
        S = []
        for i in range(n):
            if s_bin[i] == '1':
                S.append(i)

        # Figure 8 (a)
        if aux is not None:
            if j is not None:
                cls._logger.warn('With auxiliary qubit in phase_shift_s, no i_j is needed.')
            assert aux >= n, ValueError('Invalid auxiliary qubit in phase_shift_s.')
            for i in S:
                CX & [i, aux] | gates
            U1(alpha) & aux | gates
            for i in reversed(S):
                CX & [i, aux] | gates
            return gates
        # Figure 8 (b)
        else:
            assert j < len(S), ValueError('Invalid target in phase_shift without auxiliary qubit.')
            for i in S:
                if i == S[j]:
                    continue
                CX & [i, S[j]] | gates
            U1(alpha) & S[j] | gates
            for i in S:
                if i == S[j]:
                    continue
                CX & [i, S[j]] | gates
            return gates
