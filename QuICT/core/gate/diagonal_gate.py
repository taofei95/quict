import numpy as np

from QuICT.core.gate import CompositeGate


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
        self.target = target
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
            return self.no_aux_qubit()
        else:
            return self.with_aux_qubit()

    def no_aux_qubit(self):
        """
        Returns:
            CompositeGate: diagonal gate without auxiliary qubit
        """

    def with_aux_qubit(self):
        """
        Returns:
            CompositeGate: diagonal gate with auxiliary qubit at the end of qubits
        """
