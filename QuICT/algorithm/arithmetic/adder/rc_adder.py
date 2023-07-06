from numpy import pi

from QuICT.core.gate import Ry, CU3
from QuICT.core.gate.composite_gate import CompositeGate
from QuICT.algorithm.qft import ry_QFT


class RCFourierAdderWired(CompositeGate):
    """
        An adder in Fourier space using ry QFT. A quantum-classical adder
        circuit, addend is hardwired into the circuit. Assuming size of
        the quantum register is "n" and the addend is in the range, this
        adder is in place, uses n qubits.

        Based on paper "High Performance Quantum Modular Multipliers"
        by Rich Rines, Isaac Chuang: https://arxiv.org/abs/1801.01081
    """

    def __init__(
        self,
        qreg_size: int,
        addend: int,
        controlled: bool = False,
        in_fourier: bool = False,
        out_fourier: bool = False,
        name: str = None
    ):
        """
            Construct the adder circuit that adds 'addend' to a quantum
            register of size 'qreg_size'The circuit will have width
            'qreg_size' for 'controlled' is False and 'qreg_size + 1'
            for 'controlled' is True with the control bit on the index 0.

            Args:
                qreg_size (int):
                    Size of the quantum register waiting to be added. >= 2

                addend (int):
                    The integer that will add to the qreg. Can be positive
                    or negative.

                controlled (bool):
                    Indicates whether the adder is controlled by a qubit.
                    If True, a controlled adder will be constructed and
                    the 0 index is the control bit.

                in_fourier (bool):
                    If True, assuming the input register is already in ry-qft
                    basis.

                out_fourier (bool):
                    If True, after the addition, the qreg will be left in
                    ry-qft basis.

        """

        self._addend = addend
        self._controlled = controlled

        super().__init__(name)

        if controlled:
            if not in_fourier:
                ry_QFT(qreg_size) | self(range(1, qreg_size + 1))
            self._build_ctl_phi_adder(qreg_size, addend)
            if not out_fourier:
                ry_QFT(qreg_size, inverse=True) | self(range(1, qreg_size + 1))
        else:
            if not in_fourier:
                ry_QFT(qreg_size) | self
            self._build_phi_adder(qreg_size, addend)
            if not out_fourier:
                ry_QFT(qreg_size, inverse=True) | self

    @property
    def is_controlled(self):
        return self._controlled

    @property
    def addend(self):
        return self._addend

    def _build_phi_adder(
        self,
        qreg_size: int,
        addend: int
    ):
        for k in range(qreg_size):
            theta = pi * addend / (2**(k))
            Ry(theta) | self([qreg_size - 1 - k])

    def _build_ctl_phi_adder(
        self,
        qreg_size: int,
        addend: int
    ):
        for k in range(qreg_size):
            theta = pi * addend / (2**(k))
            # CU3(theta, 0, 0) is CRy(theta)
            CU3(theta, 0, 0) | self([0, qreg_size - k])
