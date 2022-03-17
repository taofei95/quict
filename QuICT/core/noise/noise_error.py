import numpy as np
from typing import List, Tuple

from QuICT.core.gate import ID, X, Y, Z
from .utils import is_kraus_ops


class QuantumNoiseError:
    def __init__(self, ops: list, num_qubits=None):
        if not isinstance(ops, (tuple, list)):
            raise TypeError("Wrong input for Noise Error.")

        if not is_kraus_ops(ops, num_qubits):
            raise KeyError("There is not a Kraus operator.")

        self.operators = ops
        self.conj_ops = self.conj_trans()

    def __str__(self):
        pass

    def conj_trans(self):
        return [np.transpose(k).conjugate() for k in self.operators]

    def compose(self, other):
        pass

    def tensor(self, other):
        pass

    def apply_to_gate(self, gate):
        return [ops.dot(gate.matrix) for ops in self.operators]


class KrausError(QuantumNoiseError):
    def __init__(self, kraus: List[np.ndarray]):
        if isinstance(kraus, np.ndarray):
            kraus = [kraus]

        super().__init__(kraus)


class UnitaryError(QuantumNoiseError):
    def __init__(self, unitaries: List[Tuple(np.ndarray, float)], num_qubits: int = 1):
        if isinstance(unitaries, (tuple, list)) and not isinstance(unitaries[0], tuple):
            raise TypeError("Wrong input for UnitaryError.")

        ops = [np.sqrt(prob) * mat for mat, prob in unitaries]
        super().__init__(ops, num_qubits)
        self._qubits = num_qubits


class PauilError(QuantumNoiseError):
    """ Pauil Error; Including Bit Flip and Phase Flip

    Args:
        QuantumNoiseError (_type_): _description_
    """
    _BASED_GATE_MATRIX = {
        'id': ID.matrix,
        'x': X.matrix,
        'y': Y.matrix,
        'z': Z.matrix
    }

    def __init__(self, ops: List[Tuple(str, float)]):
        if not isinstance(ops, (list, tuple)):
            raise TypeError("Wrong input for Pauil Error.")

        for gate_name, prob in ops:
            if (
                gate_name not in self._BASED_GATE_MATRIX or
                prob < 0 or prob > 1
            ):
                raise KeyError("Pauil Error get wrong input.")

        pauil_ops = [np.sqrt(p) * self._BASED_GATE_MATRIX[gn] for gn, p in ops]
        super().__init__(pauil_ops)


class DepolarizingError(QuantumNoiseError):
    def __init__(self, prob: float, num_qubits: int = 1):
        if not isinstance(prob, float) and not isinstance(num_qubits, int):
            raise TypeError("Wrong input for Depolarizing Error.")

        assert num_qubits >= 1, "number of qubits must be positive integer."
        num_ops = 4 ** num_qubits
        max_prob = num_ops / (num_ops - 1)
        if prob < 0 or prob > max_prob:
            raise KeyError(f"Depolarizing prob must between 0 and {max_prob}")

        probs = [1 - prob / max_prob] + [prob / num_ops] * (num_ops - 1)


class DampingError(QuantumNoiseError):
    """ Amplitude Damping, Phase Damping and Amp-Phase Damping

        a = amp, b = phase, p = state prob

        A0 = sqrt(1 - p1) * [[1, 0], [0, sqrt(1 - a - b)]]
        A1 = sqrt(1 - p1) * [[0, sqrt(a)], [0, 0]]
        A2 = sqrt(1 - p1) * [[0, 0], [0, sqrt(b)]]
        B0 = sqrt(p1) * [[sqrt(1 - a - b), 0], [0, 1]]
        B1 = sqrt(p1) * [[0, 0], [sqrt(a), 0]]
        B2 = sqrt(p1) * [[sqrt(b), 0], [0, 0]]

    Args:
        QuantumNoiseError (_type_): _description_
    """
    def __init__(self, amplitude_prob: float, phase_prob: float, mixed_state_prob: float = 0.0):
        assert amplitude_prob > 0 and amplitude_prob <= 1, f"Amplitude prob must between 0 and 1, not {amplitude_prob}."
        assert phase_prob > 0 and phase_prob <= 1, f"Phase prob must between 0 and 1, not {phase_prob}."
        assert mixed_state_prob > 0 and mixed_state_prob <= 1, f"state prob must between 0 and 1, not {mixed_state_prob}."

        if amplitude_prob + phase_prob > 1:
            raise KeyError("Invalid amplitude and phase damping prob, the sum cannot greater than 1.")

        ops = []
        if mixed_state_prob == 0:
            pass
