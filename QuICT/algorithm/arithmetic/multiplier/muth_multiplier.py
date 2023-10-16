from typing import Optional
from QuICT.core.gate import CompositeGate, CCX, Barrier
from QuICT.algorithm.arithmetic.adder import MuThCtrlAdder

from QuICT.tools.exception.core.gate_exception import GateParametersAssignedError


class MuThMultiplier(CompositeGate):
    """
        An out-of-place toffoli based cumulative quantum-quantum multiplication proposed by
        Edgard Muñoz-Coreas and Himanshu Thapliyal in "Quantum Circuit Design of a T-count
        Optimized Integer Multiplier"[1]

        [1]: https://ieeexplore.ieee.org/document/8543237
    """

    def __init__(
        self,
        qreg_size: int,
        qreg_size_b: Optional[int] = None,
        name: str = None
    ):
        """
            Construct the Muñoz-Thapliyal quantum multiplication as a composite gate:

            |a(n)>|b(m)>|0(n+m)>|0(1)> ---> |a(n)>|b(m)>|a*b(n+m)>|0(1)>

            Total requires `2 * (qreg_size + qreg_size_b) + 1` qubits.

            Args:
                qreg_size (int): Register size for the first input register

                qreg_size_b (int | None): Register size for the second input register, will be
                the same as the first input register if not given.
        """
        if qreg_size < 2:
            raise GateParametersAssignedError(f"Input register size must be larger than 1 but given {qreg_size}.")

        if qreg_size_b is None:
            qreg_size_b = qreg_size
        elif qreg_size_b < 2:
            raise GateParametersAssignedError(f"The second input register size must be larger than 1 but given {qreg_size_b}.")

        self._reg_a_list = list(range(qreg_size))
        self._reg_b_list = list(range(qreg_size, qreg_size + qreg_size_b))
        self._reg_prod_list = list(range(qreg_size + qreg_size_b, 2 * (qreg_size + qreg_size_b)))
        self._ancilla = [2 * (qreg_size + qreg_size_b)]

        super().__init__(name)

        # step 1
        for i in range(qreg_size_b):
            CCX | self([
                self._reg_a_list[-1],
                self._reg_b_list[-1 - i],
                self._reg_prod_list[-1 - i]
            ])

        # step 2 & 3
        ctrl_adder_gate = MuThCtrlAdder(qreg_size_b)
        for i in range(qreg_size - 1):
            ctrl_bit = [self._reg_a_list[-2 - i]]
            target_reg = self._reg_prod_list[qreg_size - 2 - i: qreg_size + qreg_size_b - 1 - i]
            ctrl_adder_gate | self(
                ctrl_bit + self._reg_b_list + target_reg + self._ancilla
            )
