import torch
import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import CompositeGate, Rz, sqiSwap, X
from QuICT.algorithm.quantum_machine_learning.utils import Ansatz


class Thouless:
    """ Thouless ansatz

    Reference:
        https://www.sciencedirect.com/science/article/pii/0029558260900481
        https://arxiv.org/abs/1711.04789
        https://arxiv.org/abs/2004.04174
    """
    def __init__(self, device=torch.device("cuda:0")):
        """
        Args:
            device (str, optional): The device to which the model is assigned.
                Defaults to "cuda:0".
        """
        self._device = device

    def __call__(self, orbitals, electrons, angles):
        """
        Args:
            orbitals(int): number of orbitals (i.e. quantum qubits)
            electrons(int): number of electrons
            angles(list[float]): the list of parameters

        Returns:
            Ansatz: thouless ansatz
        """
        gates = self.build_gates(orbitals, electrons, angles)
        circ = Circuit(orbitals)
        gates | circ
        ansatz = Ansatz(orbitals, circ, device=self._device)
        return ansatz

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
        Rz(np.pi) & 1 | gates
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
