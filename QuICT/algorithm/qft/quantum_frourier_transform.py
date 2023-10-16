import numpy as np

from QuICT.core.gate import H, CU1, CompositeGate


class QFT(CompositeGate):
    r""" Implement the Quantum Fourier Transform without the swap gates.

    $$
    \vert{j}\rangle \mapsto \frac{1}{2^{n/2}} \bigotimes_{l=n}^{1}[\vert{0}\rangle + e^{2\pi ij2^{-l}}\vert{1}\rangle]
    $$

    Reference:
        Nielsen, M.A., & Chuang, I.L. (2010). Quantum Computation and Quantum Information (10th Anniversary edition).

    Examples:
        >>> from QuICT.core import Circuit
        >>> from QuICT.algorithm.qft import QFT
        >>>
        >>> circuit = Circuit(3)
        >>> QFT(3) | circuit
        >>> circuit.draw(method="command", flatten=True)
                ┌───┐┌─────┐┌─────┐
        q_0: |0>┤ h ├┤ cu1 ├┤ cu1 ├─────────────────
                └───┘└──┬──┘└──┬──┘┌───┐┌─────┐
        q_1: |0>────────■──────┼───┤ h ├┤ cu1 ├─────
                               │   └───┘└──┬──┘┌───┐
        q_2: |0>───────────────■───────────■───┤ h ├
                                               └───┘
    """
    def __init__(self, targets: int, name: str = "QFT"):
        """
        Args:
            targets (int): The qubits' number.
            name (str, optional): The name of QFT gates.

        Raises:
            GateParametersAssignedError: If `targets` is smaller than 2.
        """
        assert targets >= 2, "QFT Gate need at least two targets qubits."
        super().__init__(name)
        self.qft_build(targets)

    def qft_build(self, targets):
        for i in range(targets):
            H | self(i)
            for j in range(i + 1, targets):
                CU1(2 * np.pi / (1 << j - i + 1)) | self([j, i])

    def depth(self, depth_per_qubits: bool = False):
        return 2 * self.width() - 1 if not depth_per_qubits else list(range(self.width(), 2 * self.width()))

    def inverse(self):
        inverse_gate = IQFT(self.width())
        inverse_gate & self.qubits

        return inverse_gate


class IQFT(CompositeGate):
    r""" Implement inverse of the Quantum Fourier Transform without the swap gates.

    $$
    \frac{1}{2^{n/2}} \bigotimes_{l=n}^{1}[\vert{0}\rangle + e^{2\pi ij2^{-l}}\vert{1}\rangle] \mapsto \vert{j}\rangle
    $$

    Examples:
        >>> from QuICT.core import Circuit
        >>> from QuICT.algorithm.qft import IQFT
        >>>
        >>> circuit = Circuit(3)
        >>> IQFT(3) | circuit
        >>> circuit.draw(method="command", flatten=True)
                                 ┌─────┐┌─────┐┌───┐
        q_0: |0>─────────────────┤ cu1 ├┤ cu1 ├┤ h ├
                     ┌─────┐┌───┐└──┬──┘└──┬──┘└───┘
        q_1: |0>─────┤ cu1 ├┤ h ├───┼──────■────────
                ┌───┐└──┬──┘└───┘   │
        q_2: |0>┤ h ├───■───────────■───────────────
                └───┘
    """
    def __init__(self, targets: int, name: str = "IQFT"):
        """
        Args:
            targets (int): The qubits' number.
            name (str, optional): The name of QFT gates.

        Raises:
            GateParametersAssignedError: If `targets` is smaller than 2.
        """
        assert targets >= 2, "QFT Gate need at least two targets."
        super().__init__(name)
        self.iqft_build(targets)

    def iqft_build(self, targets):
        for i in range(targets - 1, -1, -1):
            for j in range(targets - 1, i, -1):
                CU1(-2 * np.pi / (1 << j - i + 1)) | self([j, i])

            H | self(i)

    def depth(self, depth_per_qubits: bool = False):
        return 2 * self.width() - 1 if not depth_per_qubits else list(range(self.width(), 2 * self.width()))

    def inverse(self):
        inverse_gate = QFT(self.width())
        inverse_gate & self.qubits

        return inverse_gate
