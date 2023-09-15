from typing import Optional
from numpy import pi

from QuICT.core.gate import CompositeGate
from QuICT.core.gate import CU1
from QuICT.algorithm.qft import QFT, IQFT

from QuICT.tools.exception.core.gate_exception import GateParametersAssignedError


class DrapperAdder(CompositeGate):
    """
        A qft based quantum-quantum adder.

        Based on the following papers:
        "Addition on a Quantum Computer" by Thomas G. Draper[1]
        "Quantum arithmetic with the Quantum Fourier Transform" by Lidia Ruiz-Perez and
        Juan Carlos Garcia-Escartin[2]

        [1]: https://arxiv.org/abs/quant-ph/0008033v1
        [2]: https://arxiv.org/abs/1411.5949v2
    """

    def __init__(
        self,
        qreg_size: int,
        qreg_size_b: Optional[int] = None,
        in_fourier: bool = False,
        out_fourier: bool = False,
        name: str = None
    ):
        """
            Construct the quantum adder using quantum fourier transform:

            |a>|b> ---> |a>|(a+b)%(2^n)>, for n = `qreg_size`

            Total requires `2 * qreg_size` qubits.

            Args:
                qreg_size (int): Input register size for both addends. >= 2.

                in_fourier (bool): If True, assuming the input register is already in fourier
                basis. Default to be False.

                out_fourier (bool): If True, after the addition, the qreg will be left in fourier
                basis. Default to be False.
        """
        if qreg_size < 1:
            raise GateParametersAssignedError(f"Register size must be greater than or equal to 2 but given {qreg_size}")

        if qreg_size_b is None:
            qreg_size_b = qreg_size
        elif qreg_size_b < 2:
            raise GateParametersAssignedError("Input register size must be larger than 1.")

        self._reg_size = qreg_size

        self._reg_a_list = list(range(qreg_size))
        self._reg_b_list = list(range(qreg_size, qreg_size + qreg_size_b))

        super().__init__(name)

        with self:
            if not in_fourier:
                QFT(qreg_size_b) & self._reg_b_list
            # addition
            for k in range(qreg_size):
                c_bit = self._reg_a_list[qreg_size - 1 - k]
                for j in range(qreg_size_b - k):
                    t_bit = self._reg_b_list[j]
                    theta = pi / (1 << (qreg_size_b - k - 1 - j))
                    CU1(theta) & [c_bit, t_bit]
            if not out_fourier:
                IQFT(qreg_size_b) & self._reg_b_list
