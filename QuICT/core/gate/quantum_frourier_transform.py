import numpy as np

from QuICT.core.gate import H, CU1, CompositeGate


class QFT(CompositeGate):
    def __init__(self, targets: int, name: str = None):
        assert targets >= 2, "QFT Gate need at least two targets."
        qft_gates = self.qft_build(targets)
        super().__init__(name, gates=qft_gates)

    def qft_build(self, targets):
        qft_gates = []
        for i in range(targets):
            qft_gates.append(H & i)
            for j in range(i + 1, targets):
                qft_gates.append(CU1(2 * np.pi / (1 << j - i + 1)) & [j, i])

        return qft_gates

    def depth(self, depth_per_qubits: bool = False):
        return 2 * self.width() - 1 if not depth_per_qubits else list(range(self.width(), 2 * self.width()))

    def inverse(self):
        inverse_gate = IQFT(self.width())
        inverse_gate & self.qubits

        return inverse_gate


class IQFT(CompositeGate):
    def __init__(self, targets: int, name: str = None):
        assert targets >= 2, "QFT Gate need at least two targets."
        iqft_gates = self.iqft_build(targets)
        super().__init__(name, gates=iqft_gates)

    def iqft_build(self, targets):
        iqft_gates = []
        for i in range(targets - 1, -1, -1):
            for j in range(targets - 1, i, -1):
                iqft_gates.append(CU1(-2 * np.pi / (1 << j - i + 1)) & [j, i])

            iqft_gates.append(H & i)

        return iqft_gates

    def depth(self, depth_per_qubits: bool = False):
        return 2 * self.width() - 1 if not depth_per_qubits else list(range(self.width(), 2 * self.width()))

    def inverse(self):
        inverse_gate = QFT(self.width())
        inverse_gate & self.qubits

        return inverse_gate
