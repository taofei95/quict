import numpy as np

from .ansatz import Ansatz
from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate, Rz, Variable, np, sqiSwap, X


class Thouless(Ansatz):
    """ Thouless ansatz

    Reference:
        https://www.sciencedirect.com/science/article/pii/0029558260900481
        https://arxiv.org/abs/1711.04789
        https://arxiv.org/abs/2004.04174
    """
    def __init__(self, orbitals, electrons):
        """
        Args:
            orbitals(int): number of orbitals (i.e. quantum qubits)
            electrons(int): number of electrons
        """
        super(Thouless, self).__init__(orbitals)
        self.orbitals = orbitals
        self.electrons = electrons

    def init_circuit(self, angles=None):
        """
        Args:
            angles(Variable/np.ndarray): the list of parameters

        Returns:
            Circuit: thouless ansatz
        """
        if angles is None:
            angles = np.zeros(self.electrons * (self.orbitals - self.electrons))
        angles = Variable(pargs=angles) if isinstance(angles, np.ndarray) else angles
        assert angles.shape == (self.electrons * (self.orbitals - self.electrons),), \
            ValueError("Incorrect number of parameters")
        self._params = angles

        circuit = Circuit(self.orbitals)
        # The first X gates
        for k in range(self.electrons):
            X | circuit(k)

        # Givens rotations
        param = 0
        for layer in range(self.orbitals):
            for k in range(
                abs(self.electrons - layer),
                self.orbitals - abs(self.orbitals - (self.electrons + layer)),
                2
            ):
                sqiSwap | circuit([k, k + 1])
                Rz(0 - self._params[param]) | circuit(k)
                Rz(np.pi + self._params[param]) | circuit(k + 1)
                sqiSwap | circuit([k, k + 1])
                Rz(-np.pi) | circuit(k + 1)
                param += 1

        return circuit

    @staticmethod
    def modified_Givens_rotation(angle):
        """
        Args:
            angle(float): the rotation angle of the Rz gate
        """
        gates = CompositeGate()
        sqiSwap & [0, 1] | gates
        Rz(-angle) & 0 | gates
        Rz(np.pi + angle) & 1 | gates
        sqiSwap & [0, 1] | gates
        Rz(-np.pi) & 1 | gates
        return gates

    @staticmethod
    def build_gates(orbitals, electrons, angles):
        """
        Args:
            orbitals(int): number of orbitals (i.e. quantum qubits)
            electrons(int): number of electrons
            angles(list[float]): the list of parameters

        Returns:
            CompositeGate: thouless ansatz
        """
        assert len(angles) == electrons * (orbitals - electrons), ValueError("Incorrect number of parameters")
        ansatz = CompositeGate()

        # The first X gates
        for k in range(electrons):
            X & k | ansatz

        # Givens rotations
        param = 0
        for layer in range(orbitals):
            for k in range(
                abs(electrons - layer),
                orbitals - abs(orbitals - (electrons + layer)),
                2
            ):
                rot = Thouless.modified_Givens_rotation(angles[param])
                rot | ansatz([k, k + 1])
                param += 1

        return ansatz
