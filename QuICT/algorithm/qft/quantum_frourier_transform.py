import numpy as np

from QuICT.core.gate import H, CU1, CompositeGate
from QuICT.tools.exception.core.gate_exception import GateParametersAssignedError


class QFT(CompositeGate):
    """ Implement a QFT Gates. """
    def __init__(self, targets: int, name: str = "QFT"):
        """
        Args:
            targets (int): The qubits' number.
            name (str, optional): The name of QFT gates. Defaults to None.
        """
        if targets < 2:
            raise GateParametersAssignedError("QFT Gate needs at least two target qubits.")

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
    """ Implement an IQFT Gates. """
    def __init__(self, targets: int, name: str = "IQFT"):
        """
        Args:
            targets (int): The qubits' number.
            name (str, optional): The name of QFT gates. Defaults to None.
        """
        if targets < 2:
            raise GateParametersAssignedError("IQFT Gate needs at least two target qubits.")

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
