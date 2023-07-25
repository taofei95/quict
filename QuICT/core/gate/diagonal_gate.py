import numpy as np

from QuICT.core.gate import CompositeGate, CX, Rz, U1
from QuICT.tools import Logger


_logger = Logger("DiagonalGate")


class DiagonalGate(object):
    """
    Diagonal gate

    Reference:
        https://arxiv.org/abs/2108.06150
    """
    def __init__(self, target: int, aux: int = 0):
        """
        Args:
            target(int): number of target qubits
            aux(int, optional): number of auxiliary qubits
        """
        self._logger = _logger
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

    def phase_shift(self, s, alpha, aux=None, j=None):
        """
        Implement the phase shift defined in Equation 5 as Figure 8
        |x> -> exp(i alpha_s <s, x>) |x>

        Args:
            s(int): whose binary representation stands for the 0-1 string s
            alpha(float): alpha_s in the equation
            aux(int, optional): key of auxiliary qubit (if exists)
            j(int, optional): if no auxiliary qubit, the j-th smallest element in S would be the target qubit

        Returns:
            CompositeGate: CompositeGate for Equation 5 as Figure 8
        """
        assert 1 <= s <= 1 << self.target, ValueError('Invalid controller in phase_shift.')
        gates = CompositeGate()
        s_bin = np.binary_repr(s, width=self.target)
        S = []
        for i in range(self.target):
            if s_bin[i] == '1':
                S.append(i)

        # Figure 8 (a)
        if aux is not None:
            if j is not None:
                self._logger.warn('With auxiliary qubit in phase_shift, no i_j is needed.')
            assert aux >= self.target, ValueError('Invalid auxiliary qubit in phase_shift.')
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
