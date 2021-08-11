"""
Encoders transform ladder operators to CompositeGate on zero state of qubits, 
while decoders transform statevector of qubits to FermionOperator on vacuum state.
"""

from abc import ABC, abstractclassmethod
import numpy as np

class Encoder(ABC):
    """
    Abstract class of encoding methods, which contains the pairs of encoder and decoder.
    """
    @abstractclassmethod
    def encoder(cls, operator):
        """
        Encoders transform ladder operators to CompositeGate on zero state of qubits

        Args:
            operator(FermionOperator): FermionOperator to be transformed

        Returns:
            CompositeGate: The corresponding CompositeGate on zero state of qubits
        """
        pass

    @abstractclassmethod
    def decoder(cls, prob):
        """
        Decoders transform statevector of qubits to ladder operators on vacuum state

        Args:
            prob(np.ndarray): Probability array of the statevector of qubits

        Returns:
            FermionOperator: The corresponding FermionOperator on vacuum state
        """
        pass

class JordanWigner(Encoder):
    """
    Implement the Jordan-Wigner encoding method
    """
    @classmethod
    def encoder(cls, operator):
        pass

    @classmethod
    def decoder(cls, prob):
        pass

class Parity(Encoder):
    """
    Implement the parity encoding method
    """
    @classmethod
    def encoder(cls, operator):
        pass

    @classmethod
    def decoder(cls, prob):
        pass

class BravyiKitaev(Encoder):
    """
    Implement the Bravyi-Kitaev encoding method
    """
    @classmethod
    def encoder(cls, operator):
        pass

    @classmethod
    def decoder(cls, prob):
        pass
