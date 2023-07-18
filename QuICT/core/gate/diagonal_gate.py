import numpy as np

from QuICT.core.gate import CompositeGate, CX
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
        assert len(angles) == 2 ** self.target, ValueError('Incorrect number of angles')
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
        n = self.target
        m = self.aux
        gates = CompositeGate()

        # Stage 1: Prefix Copy
        t = np.floor(np.log2(m / 2))
        copies = np.floor(m / (2 * t))
        for j in range(copies):
            for i in range(t):
                CX & [i, n + i * copies + j] | gates

        # Stage 2: Gray Initial
