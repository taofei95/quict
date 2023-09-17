from numpy import pi

from QuICT.core.gate import CompositeGate, CU1
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
        if qreg_size < 2:
            raise GateParametersAssignedError(f"Register size must be greater than or equal to 2 but given {qreg_size}")

        self._reg_size = qreg_size

        self._reg_a_list = list(range(qreg_size))
        self._reg_b_list = list(range(qreg_size, 2 * qreg_size))

        super().__init__(name)

        if not in_fourier:
            QFT(qreg_size) | self(self._reg_b_list)
        # addition
        for k in reversed(range(qreg_size)):
            for j in range(k + 1):
                theta = pi / (1 << (k - j))
                CU1(theta) | self([k, qreg_size + j])
        if not out_fourier:
            IQFT(qreg_size) | self(self._reg_b_list)
