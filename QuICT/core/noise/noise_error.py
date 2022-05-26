from itertools import product
import math
import numpy as np
from typing import List, Tuple

from QuICT.ops.linalg.cpu_calculator import tensor, dot
from QuICT.core.gate import ID, X, Y, Z
from .utils import is_kraus_ops, NoiseChannel


class QuantumNoiseError:
    @property
    def kraus_operators(self):
        return self._operators

    @property
    def kraus_operators_ctranspose(self):
        return self._conj_ops

    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, channel: NoiseChannel):
        assert isinstance(channel, NoiseChannel)
        self._type = channel

    @property
    def qubits(self):
        return self._qubits

    def __init__(self, ops: list):
        if not isinstance(ops, (tuple, list)):
            raise TypeError("Wrong input for Noise Error.")

        if not is_kraus_ops(ops):
            raise KeyError("There is not a Kraus operator.")

        self._operators = ops
        self._conj_ops = self.conj_trans()
        self._qubits = int(np.log2(ops[0].shape[0]))
        self._type = NoiseChannel.kraus

    def __str__(self):
        ne_str = f"{self.type.value} with {self.qubits} qubits.\n"
        for i, km in enumerate(self.kraus_operators):
            ne_str += f"Kraus_{i}: {km}\n"

        return ne_str

    def conj_trans(self):
        return [np.transpose(k).conjugate() for k in self.kraus_operators]

    def compose(self, other):
        assert other.qubits == self.qubits

        ops = [dot(so, oo) for oo in other.kraus_operators for so in self.kraus_operators]
        return KrausError(ops)

    def tensor(self, other):
        assert other.qubits == self.qubits

        ops = [tensor(so, oo) for oo in other.kraus_operators for so in self.kraus_operators]
        return KrausError(ops)

    def apply_to_gate(self, matrix: np.ndarray):
        assert matrix.shape == (2 ** self._qubits, 2 ** self._qubits)

        return [ops.dot(matrix) for ops in self.kraus_operators]


class KrausError(QuantumNoiseError):
    def __init__(self, kraus: List[np.ndarray]):
        if isinstance(kraus, np.ndarray):
            kraus = [kraus]

        super().__init__(kraus)


class UnitaryError(QuantumNoiseError):
    def __init__(self, unitaries: List[Tuple[np.ndarray, float]]):
        if isinstance(unitaries, (tuple, list)) and not isinstance(unitaries[0], tuple):
            raise TypeError("Wrong input for Unitary Error.")

        if sum([prob for _, prob in unitaries]) != 1:
            raise KeyError("The sum of probability of unitary matrix should be 1.")

        ops = [np.sqrt(prob) * mat for mat, prob in unitaries]
        super().__init__(ops)
        self.type = NoiseChannel.unitary


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

    def __init__(self, ops: List[Tuple[str, float]], num_qubits: int = 1):
        if not isinstance(ops, (list, tuple)):
            raise TypeError("Wrong input for Pauil Error.")

        total_prob = 0
        for op, prob in ops:
            if prob < 0 or prob > 1 or (len(op) != num_qubits and num_qubits > 1):
                raise KeyError("Pauil Error get wrong input.")

            for g in op:
                if g not in list(self._BASED_GATE_MATRIX.keys()):
                    raise KeyError("Pauil Error get wrong input.")

            total_prob += prob

        if not math.isclose(total_prob, 1, rel_tol=1e-6):
            raise KeyError("Pauil Error get wrong input.")

        super().__init__(self._create_kraus_ops(ops, num_qubits))
        self.pauil_ops = ops
        self.type = NoiseChannel.pauil

    def __str__(self):
        pn_str = f"{self.type.value} with {self.qubits} qubits.\n"
        for gate_str, prob in self.pauil_ops:
            pn_str += f"[{gate_str}, {prob}] "

        return pn_str

    def _create_kraus_ops(self, pauil_ops, num_qubits) -> list:
        num_qubit = len(pauil_ops[0][0])
        kraus_ops = []

        for pauil_op, prob in pauil_ops:
            if len(pauil_op) != num_qubit:
                raise KeyError("Input should has same size for Pauil Error.")

            based_matrix = self._BASED_GATE_MATRIX[pauil_op[0]]
            for g in pauil_op[1:]:
                if num_qubits > 1:
                    based_matrix = tensor(based_matrix, self._BASED_GATE_MATRIX[g])
                else:
                    based_matrix = dot(based_matrix, self._BASED_GATE_MATRIX[g])

            kraus_ops.append(np.sqrt(prob) * based_matrix)

        return kraus_ops


class BitflipError(PauilError):
    def __init__(self, prob: float):
        assert prob >= 0 and prob <= 1, "Wrong input for Bitflip Error."
        ops = [('i', 1 - prob), ('x', prob)]

        super().__init__(ops)


class PhaseflipError(PauilError):
    def __init__(self, prob: float):
        assert prob >= 0 and prob <= 1, "Wrong input for Phaseflip Error."
        ops = [('i', 1 - prob), ('z', prob)]

        super().__init__(ops)


class PhaseBitflipError(PauilError):
    def __init__(self, prob: float):
        assert prob >= 0 and prob <= 1, "Wrong input for PhaseBitflip Error."
        ops = [('i', 1 - prob), ('y', prob)]

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

        super().__init__(list(zip(ops, probs)), num_qubits)
        self.type = NoiseChannel.depolarizing


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
    def __init__(self, amplitude_prob: float, phase_prob: float, dissipation_state: float = 0.0):
        assert amplitude_prob >= 0 and amplitude_prob <= 1, \
            f"Amplitude prob must between 0 and 1, not {amplitude_prob}."
        assert phase_prob >= 0 and phase_prob <= 1, f"Phase prob must between 0 and 1, not {phase_prob}."
        assert dissipation_state >= 0 and dissipation_state <= 1, \
            f"state prob must between 0 and 1, not {dissipation_state}."

        if amplitude_prob + phase_prob > 1:
            raise KeyError("Invalid amplitude and phase damping prob, the sum cannot greater than 1.")

        self.amplitude_prob = amplitude_prob
        self.phase_prob = phase_prob
        self.dissipation_state = dissipation_state

        super().__init__(self._create_kraus_ops())
        self.type = NoiseChannel.damping

    def __str__(self):
        damp_noise = f"{self.type.value} with 1 qubit\n"
        if self.amplitude_prob != 0:
            damp_noise += f"amplitude damping prob: {self.amplitude_prob} "

        if self.phase_prob != 0:
            damp_noise += f"phase damping prob: {self.phase_prob} "

        if self.dissipation_state != 0:
            damp_noise += f"prob of stable state: {self.dissipation_state}"

        return damp_noise

    def _create_kraus_ops(self):
        m0, m1 = np.sqrt(1 - self.dissipation_state), np.sqrt(self.dissipation_state)
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
        return [kraus[nzi] for nzi in non_zero_idxes]
