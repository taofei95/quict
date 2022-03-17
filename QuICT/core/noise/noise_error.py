import numpy as np
from typing import List, Tuple

from QuICT.core.utils import GateType


class QuantumNoiseError:
    def __init__(self):
        pass
    
    def __str__(self):
        pass
    
    def compose(self):
        pass
    
    def tensor(self):
        pass

    def apply_to_gate(self, gate):
        pass


class KrausError(QuantumNoiseError):
    def __init__(self, kraus: List[np.ndarray]):
        pass


class PauilError(QuantumNoiseError):
    """ Pauil Error; Including Bit Flip and Phase Flip

    Args:
        QuantumNoiseError (_type_): _description_
    """
    _BASED_GATE = [GateType.id, GateType.x, GateType.y, GateType.z]
    
    def __init__(self, ops: List[Tuple(str, float)]):
        pass


class DepolaringError(QuantumNoiseError):
    def __init__(self, prob: float, num_qubits: int = 1):
        pass


class DampingError(QuantumNoiseError):
    """ Amplitude Damping, Phase Damping and Amp-Phase Damping

    Args:
        QuantumNoiseError (_type_): _description_
    """
    def __init__(self, amplitude_prob: float, phase_prob: float, mixed_state_prob: float = 1.0):
        pass
    

class UnitaryError(QuantumNoiseError):
    def __init__(self, ops: List[Tuple(np.ndarray, float)], num_qubits: int = 1):
        pass
