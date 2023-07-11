import numpy as np

from QuICT.core.gate import CU3, CompositeGate


class ry_QFT(CompositeGate):
    """
        Quantum Fourier Transform using Ry rotations instead of Rz, H
        (and swap).

        Based on paper "High Performance Quantum Modular Multipliers"
        by Rich Rines, Isaac Chuang: https://arxiv.org/abs/1801.01081
    """

    def __init__(
        self,
        targets: int,
        approx_level: int = 0,
        name: str = None
    ):
        """
            Args:
                targets (int):
                    Qubits number

                aprrox_level (int):
                    For approx_level > 0, all rotation angles <
                    pi/(2**approx_level) will be omitted.When approx_level = 0,
                    no approximation and the qft will be exact.

                name (string):
                    Name of the gate.
        """
        assert targets >= 2, "ryQFT Gate need at least two targets."
        assert approx_level >= 0, "approximation level must be non-negative"

        self._approx_level = approx_level

        super().__init__(name)

        self._build_ry_qft(targets, approx_level)

    @property
    def approx_level(self):
        """
            an integer represents the approximation level of the qft circuit.
        """
        return self._approx_level

    def _build_ry_qft(self, targets: int, approx_level: int):
        # construct qft
        for ctl_q in range(1, targets):
            for target_q in range(ctl_q):
                angle_exp = ctl_q - target_q
                # check if the approx_level is set
                if approx_level > 0 and angle_exp > approx_level:
                    continue
                theta = np.pi / (2**angle_exp)
                # CU3(theta, 0, 0) is CRy(theta)
                CU3(theta, 0, 0) | self([ctl_q, target_q])


class ry_IQFT(CompositeGate):
    """
        An inversed Ry quantum fourier transform
    """
    def __init__(
        self,
        targets: int,
        approx_level: int = 0,
        name: str = None
    ):
        """
            Args:
                targets (int):
                    Qubits number

                aprrox_level (int):
                    For approx_level > 0, all rotation angles <
                    pi/(2**approx_level) will be omitted.When approx_level = 0,
                    no approximation and the qft will be exact.

                name (string):
                    Name of the gate.
        """

        assert targets >= 2, "ryIQFT Gate need at least two targets."
        assert approx_level >= 0, "approximation level must be non-negative"

        self._approx_level = approx_level

        super().__init__(name)

        self._build_ry_iqft(targets, approx_level)

    @property
    def approx_level(self):
        """
            an integer represents the approximation level of the qft circuit.
        """
        return self._approx_level

    def _build_ry_iqft(self, targets: int, approx_level: int):
        # construct iqft
        for ctl_q in reversed(range(1, targets)):
            for target_q in reversed(range(ctl_q)):
                angle_exp = ctl_q - target_q
                # check if the approx_level is set
                if approx_level > 0 and angle_exp > approx_level:
                    continue
                theta = np.pi / (2**angle_exp)
                CU3(-theta, 0, 0) | self([ctl_q, target_q])
