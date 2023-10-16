from typing import List, Optional
from numpy import pi

from QuICT.core.gate import CompositeGate, CU1, CCX
from QuICT.algorithm.qft import QFT, IQFT

from QuICT.tools.exception.core import GateParametersAssignedError


class RGMultiplier(CompositeGate):
    """
        A qft based out-of-place quantum-quantum multiplier proposed by Lidia Ruiz-Perez and
        Juan Carlos Garcia-Escartin in "Quantum arithmetic with the Quantum Fourier Transform"[1]

        [1]: https://arxiv.org/abs/1411.5949v2
    """

    def __init__(
        self,
        qreg_size: int,
        qreg_size_b: Optional[int] = None,
        name: str = None
    ):
        """
            Construct a quantum-quantum multiplier using quantum fourier transform:

            |a(n)>|b(m)>|0(n+m)> ---> |a(n)>|b(m)>|a*b(n+m)>

            Total requires `2 * (qreg_size + qreg_size_b)` qubits.

            Args:
                qreg_size (int): Register size for the first input register

                qreg_size_b (int | None): Register size for the second input register, will be
                the same as the first input register if not given.
        """
        if qreg_size < 2:
            raise GateParametersAssignedError(f"Input register size must be larger than but given {qreg_size}.")

        if qreg_size_b is None:
            qreg_size_b = qreg_size
        elif qreg_size_b < 2:
            raise GateParametersAssignedError(f"The second input register size must be larger than 1 but given {qreg_size_b}.")

        self._reg_a_list = list(range(qreg_size))
        self._reg_b_list = list(range(qreg_size, qreg_size + qreg_size_b))
        self._reg_prod_list = list(range(qreg_size + qreg_size_b, 2 * (qreg_size + qreg_size_b)))

        super().__init__(name)

        # construct circuit
        QFT(qreg_size + qreg_size_b) | self(self._reg_prod_list)
        # cumulatively add 'b << i' to result register controlled by a's i_th bit
        for i in range(qreg_size):
            self._build_ctrl_phi_shift_adder(
                reg_size_a=qreg_size_b,
                reg_size_b=qreg_size + qreg_size_b - i
            ) | self([qreg_size - 1 - i] + self._reg_b_list + self._reg_prod_list[:qreg_size + qreg_size_b - i])

        IQFT(qreg_size + qreg_size_b) | self(self._reg_prod_list)

    def _build_CCU1(self, theta) -> CompositeGate:
        """
            Construct a doubly controlled U1 gate by given rotation angle theta.
        """
        CCU1 = CompositeGate("CCU1")

        CU1(theta / 2) | CCU1([0, 1])
        CCX | CCU1([0, 1, 2])
        CU1(-theta / 2) | CCU1([0, 2])
        CCX | CCU1([0, 1, 2])
        CU1(theta / 2) | CCU1([0, 2])

        return CCU1

    def _build_ctrl_phi_shift_adder(self, reg_size_a, reg_size_b) -> CompositeGate:
        """
            A controlled adder that add (a << shift) to b in fourier space assuming both registers
            already in qft basis.

            |c>|a>|b> ---> |c>|a>|b + c * (a << shift)>

            Circuit width: `1 + reg_size_a + reg_size_b`
        """
        c_adder = CompositeGate("cAdder")

        ctrl = 0
        reg_a = list(range(1, 1 + reg_size_a))
        reg_b = list(range(1 + reg_size_a, 1 + reg_size_a + reg_size_b))

        for k in range(reg_size_a):
            for j in range(reg_size_b - k):
                theta = pi / (1 << (reg_size_b - k - 1 - j))
                self._build_CCU1(theta) | c_adder([ctrl, reg_a[-1 - k], reg_b[j]])

        return c_adder
