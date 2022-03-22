from itertools import product
import numpy as np
from typing import List, Tuple

from QuICT.ops.linalg.cpu_calculator import tensor, dot
from QuICT.core.gate import ID, X, Y, Z
from .utils import is_kraus_ops


class QuantumNoiseError:
    def __init__(self, ops: list):
        if not isinstance(ops, (tuple, list)):
            raise TypeError("Wrong input for Noise Error.")

        if not is_kraus_ops(ops):
            raise KeyError("There is not a Kraus operator.")

        self.operators = ops
        self.conj_ops = self.conj_trans()
        self.qubits = int(np.log2(ops[0].shape[0]))

    def __str__(self):
        pass

    def conj_trans(self):
        return [np.transpose(k).conjugate() for k in self.operators]

    def compose(self, other):
        assert other.qubits == self.qubits

        ops = [dot(so, oo) for oo in other.operators for so in self.operators]
        return KrausError(ops)

    def tensor(self, other):
        assert other.qubits == self.qubits

        ops = [tensor(so, oo) for oo in other.operators for so in self.operators]
        return KrausError(ops)

    def apply_to_gate(self, gate):
        return [ops.dot(gate.matrix) for ops in self.operators]


class KrausError(QuantumNoiseError):
    def __init__(self, kraus: List[np.ndarray]):
        if isinstance(kraus, np.ndarray):
            kraus = [kraus]

        super().__init__(kraus)


class UnitaryError(QuantumNoiseError):
    def __init__(self, unitaries: List[Tuple(np.ndarray, float)]):
        if isinstance(unitaries, (tuple, list)) and not isinstance(unitaries[0], tuple):
            raise TypeError("Wrong input for Unitary Error.")

        if sum([prob for _, prob in unitaries]) != 1:
            raise KeyError("The sum of probability of unitary matrix should be 1.")        

        ops = [np.sqrt(prob) * mat for mat, prob in unitaries]
        super().__init__(ops)


class PauilError(QuantumNoiseError):
    """ Pauil Error; Including Bit Flip and Phase Flip

    Example:
        PauilError([('i', 0.3), ('x', 0.7)])

    Args:
        QuantumNoiseError (_type_): _description_
    """
    _BASED_GATE_MATRIX = {
        'i': ID.matrix,
        'x': X.matrix,
        'y': Y.matrix,
        'z': Z.matrix
    }

    def __init__(self, ops: List[Tuple(str, float)]):
        if not isinstance(ops, (list, tuple)):
            raise TypeError("Wrong input for Pauil Error.")

        total_prob = 0
        for op, prob in ops:
            if prob < 0 or prob > 1:
                raise KeyError("Pauil Error get wrong input.")

            for g in op:
                if g not in list(self._BASED_GATE_MATRIX.keys()):
                    raise KeyError("Pauil Error get wrong input.")

            total_prob += prob

        if total_prob != 1:
            raise KeyError("Pauil Error get wrong input.")

        self.operators, self.qubits = self._create_kraus_ops(ops)
        self.conj_ops = self.conj_trans()

    def _create_kraus_ops(self, pauil_ops):
        num_qubit = len(pauil_ops[0][0])
        kraus_ops = []

        for pauil_op, prob in pauil_ops:
            if len(pauil_op) != num_qubit:
                raise KeyError("Input should has same size for Pauil Error.")

            based_matrix = self._BASED_GATE_MATRIX[pauil_op[0]]
            for g in pauil_op[1:]:
                based_matrix = tensor(based_matrix, self._BASED_GATE_MATRIX[g])

            kraus_ops.append(prob * based_matrix)

        return kraus_ops, num_qubit


class BitflipError(PauilError):
    def __init__(self, prob: float):
        assert prob >= 0 and prob <= 1, "Wrong input for Bitflip Error."
        ops = [('i', prob), ('x', 1 - prob)]

        super().__init__(ops)


class PhaseflipError(PauilError):
    def __init__(self, prob: float):
        assert prob >= 0 and prob <= 1, "Wrong input for Phaseflip Error."
        ops = [('i', prob), ('z', 1 - prob)]

        super().__init__(ops)


class PhaseBitflipError(PauilError):
    def __init__(self, prob: float):
        assert prob >= 0 and prob <= 1, "Wrong input for PhaseBitflip Error."
        ops = [('i', prob), ('y', 1 - prob)]

        super().__init__(ops)


class DepolarizingError(PauilError):
    def __init__(self, prob: float, num_qubits: int = 1):
        if not isinstance(prob, float) and not isinstance(num_qubits, int):
            raise TypeError("Wrong input for Depolarizing Error.")

        assert num_qubits >= 1, "number of qubits must be positive integer."
        num_ops = 4 ** num_qubits
        max_prob = num_ops / (num_ops - 1)
        if prob < 0 or prob > max_prob:
            raise KeyError(f"Depolarizing prob must between 0 and {max_prob}")

        probs = [1 - prob / max_prob] + [prob / num_ops] * (num_ops - 1)
        ops = [''.join(i) for i in product(self._BASED_GATE_MATRIX.keys(), repeat=num_qubits)]

        super().__init__(list(zip(ops, probs)))


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
        assert amplitude_prob >= 0 and amplitude_prob <= 1, f"Amplitude prob must between 0 and 1, not {amplitude_prob}."
        assert phase_prob >= 0 and phase_prob <= 1, f"Phase prob must between 0 and 1, not {phase_prob}."
        assert mixed_state_prob >= 0 and mixed_state_prob <= 1, f"state prob must between 0 and 1, not {mixed_state_prob}."

        if amplitude_prob + phase_prob > 1:
            raise KeyError("Invalid amplitude and phase damping prob, the sum cannot greater than 1.")

        self.amplitude_prob = amplitude_prob
        self.phase_prob = phase_prob
        self.mixed_state_prob = mixed_state_prob

        ops = self._create_kraus_ops()
        super().__init__(ops)

    def _create_kraus_ops(self):
        m0, m1 = np.sqrt(1 - self.mixed_state_prob), np.sqrt(self.mixed_state_prob)
        sqrt_amp, sqrt_phase = np.sqrt(self.amplitude_prob), np.sqrt(self.phase_prob)
        sqrt_ap = np.sqrt(1 - self.amplitude_prob - self.phase_prob)

        k0 = m0 * np.array([[1, 0], [0, sqrt_ap]], dtype=np.complex128)
        k1 = m0 * np.array([[0, sqrt_amp], [0, 0]], dtype=np.complex128)
        k2 = m0 * np.array([[0, 0], [0, sqrt_phase]], dtype=np.complex128)
        k3 = m1 * np.array([[sqrt_ap, 0], [0, 1]], dtype=np.complex128)
        k4 = m1 * np.array([[0, 0], [sqrt_amp, 0]], dtype=np.complex128)
        k5 = m1 * np.array([[sqrt_phase, 0], [0, 0]], dtype=np.complex128)
        kraus = [k0, k1, k2, k3, k4, k5]

        probs = [a * b for a, b in product([m0, m1], [1, sqrt_amp, sqrt_phase])]
        non_zero_idxes = [idx for idx, prob in enumerate(probs) if prob != 0]
        return kraus[non_zero_idxes]
