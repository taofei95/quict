"""
Encoders transform ladder operators to CompositeGate on zero state of qubits.
"""

from abc import ABC, abstractclassmethod
import numpy as np

class Encoder(ABC):
    """
    Abstract class of encoding methods.
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

class JordanWigner(Encoder):
    """
    Implement the Jordan-Wigner encoding method
    """
    @classmethod
    def encoder(cls, operator):
        pass

class Parity(Encoder):
    """
    Implement the parity encoding method
    """
    @classmethod
    def encoder(cls, operator):
        pass

class BravyiKitaev(Encoder):
    """
    Implement the Bravyi-Kitaev encoding method
    """
    @classmethod
    def encoder(cls, operator):
        pass
