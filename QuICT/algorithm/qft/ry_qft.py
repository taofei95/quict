import numpy as np

from QuICT.core.gate import CU3, CompositeGate


class ry_QFT(CompositeGate):
    r"""
        Quantum Fourier Transform using $R_y$ rotations instead of $R_z$ and $H$.

        $$
        \vert{j}\rangle \mapsto \bigotimes_{l=n}^{1}[R_{y}(2\pi j2^{-l})\vert{0}\rangle]
        $$

        Reference:
            "High Performance Quantum Modular Multipliers" by Rich Rines, Isaac Chuang
            <https://arxiv.org/abs/1801.01081>

        Examples:
            >>> from QuICT.core import Circuit
            >>> from QuICT.algorithm.qft import ry_QFT
            >>> # An exact ryQFT on 4 qubits
            >>> circuit = Circuit(4)
            >>> ry_QFT(4) | circuit
            >>> circuit.draw(method="command", flatten=True)
                    ┌─────┐┌─────┐       ┌─────┐
            q_0: |0>┤ cry ├┤ cry ├───────┤ cry ├──────────────
                    └──┬──┘└──┬──┘┌─────┐└──┬──┘┌─────┐
            q_1: |0>───■──────┼───┤ cry ├───┼───┤ cry ├───────
                              │   └──┬──┘   │   └──┬──┘┌─────┐
            q_2: |0>──────────■──────■──────┼──────┼───┤ cry ├
                                            │      │   └──┬──┘
            q_3: |0>────────────────────────■──────■──────■───
            >>> # An approximate ryQFT on 4 qubits with approx_level equal to 2
            >>> circuit = Circuit(4)
            >>> ry_QFT(4, approx_level=2) | circuit
            >>> circuit.draw(method="command", flatten=True)
                    ┌─────┐┌─────┐
            q_0: |0>┤ cry ├┤ cry ├─────────────────────
                    └──┬──┘└──┬──┘┌─────┐┌─────┐
            q_1: |0>───■──────┼───┤ cry ├┤ cry ├───────
                              │   └──┬──┘└──┬──┘┌─────┐
            q_2: |0>──────────■──────■──────┼───┤ cry ├
                                            │   └──┬──┘
            q_3: |0>────────────────────────■──────■───

    """

    def __init__(
        self,
        targets: int,
        approx_level: int = 0,
        name: str = "ryQFT"
    ):
        r"""
            Args:
                targets (int): Qubits number.

                approx_level (int):
                    For `approx_level` greater than 0, there will be at most `approx_level`
                    number of rotation gates on each qubit. `approx_level = 0` means no approximation.

                name (string): Name of the gate.

            Raises:
                GateParametersAssignedError: If `targets` is smaller than 2 or `approx_level` is negative.
        """
        assert targets >= 2, "ryQFT Gate need at least two targets."
        assert approx_level >= 0, "approximation level must be non-negative"

        self._approx_level = approx_level

        super().__init__(name)

        self._build_ry_qft(targets, approx_level)

    @property
    def approx_level(self):
        """ The approximation level of the qft circuit. """
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
    r"""
        An inversed Ry quantum fourier transform.

        $$
        \bigotimes_{l=n}^{1}[R_{y}(2\pi j2^{-l})\vert{0}\rangle] \mapsto  \vert{j}\rangle
        $$

        Examples:
            >>> from QuICT.core import Circuit
            >>> from QuICT.algorithm.qft import ry_IQFT
            >>> # An exact inverse ryQFT on 4 qubits
            >>> circuit = Circuit(4)
            >>> ry_IQFT(4) | circuit
            >>> circuit.draw(method="command", flatten=True)
                                  ┌─────┐       ┌─────┐┌─────┐
            q_0: |0>──────────────┤ cry ├───────┤ cry ├┤ cry ├
                           ┌─────┐└──┬──┘┌─────┐└──┬──┘└──┬──┘
            q_1: |0>───────┤ cry ├───┼───┤ cry ├───┼──────■───
                    ┌─────┐└──┬──┘   │   └──┬──┘   │
            q_2: |0>┤ cry ├───┼──────┼──────■──────■──────────
                    └──┬──┘   │      │
            q_3: |0>───■──────■──────■────────────────────────
            >>> # An approximate inverse ryQFT on 4 qubits with approx_level equal to 2
            >>> circuit = Circuit(4)
            >>> ry_IQFT(4, approx_level=2) | circuit
            >>> circuit.draw(method="command", flatten=True)
                                         ┌─────┐┌─────┐
            q_0: |0>─────────────────────┤ cry ├┤ cry ├
                           ┌─────┐┌─────┐└──┬──┘└──┬──┘
            q_1: |0>───────┤ cry ├┤ cry ├───┼──────■───
                    ┌─────┐└──┬──┘└──┬──┘   │
            q_2: |0>┤ cry ├───┼──────■──────■──────────
                    └──┬──┘   │
            q_3: |0>───■──────■────────────────────────

    """
    def __init__(
        self,
        targets: int,
        approx_level: int = 0,
        name: str = "ryIQFT"
    ):
        r"""
            Args:
                targets (int): Qubits number.

                approx_level (int):
                    For `approx_level` greater than 0, there will be at most `approx_level`
                    number of rotation gates on each qubit. `approx_level = 0` means no approximation.

                name (string): Name of the gate.

            Raises:
                GateParametersAssignedError: If `targets` is smaller than 2 or `approx_level` is negative.
        """

        assert targets >= 2, "ryIQFT Gate need at least two targets."
        assert approx_level >= 0, "approximation level must be non-negative"

        self._approx_level = approx_level

        super().__init__(name)

        self._build_ry_iqft(targets, approx_level)

    @property
    def approx_level(self):
        """ The approximation level of the inverse qft circuit. """
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
