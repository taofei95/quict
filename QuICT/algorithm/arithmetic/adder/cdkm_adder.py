from QuICT.core.gate import CompositeGate
from QuICT.core.gate import CX
from QuICT.algorithm.arithmetic.adder.utils import Maj, UnMaj


class CDKMAdder(CompositeGate):
    """
        A toffoli based quantum-quantum adder with an input carry. Based on "A new quantum
        ripple-carry addition circuit" by Steven A. Cuccaro, Thomas G. Draper, Samuel A. Kutin
        and David Petrie Moulton.[1]

        [1]: https://arxiv.org/abs/quant-ph/0410184v1
    """
    def __init__(
        self,
        qreg_size: int,
        name: str = None
    ):
        """
            Construct the quantum adder:

            |carry>|a>|b>|0> ---> |carry>|a>|a+b>|0>
        """

        super().__init__(name)

        natrual_gate = CompositeGate()
        maj = Maj()
        uma = UnMaj()
        with natrual_gate:
            for i in range(qreg_size):
                maj & [2 * i, 2 * i + 1, 2 * i + 2]
            CX & [2 * qreg_size, 2 * qreg_size + 1]
            for i in reversed(range(qreg_size)):
                uma & [2 * i, 2 * i + 1, 2 * i + 2]

        q_list = [0]
        for i in range(qreg_size):
            q_list += [2 * qreg_size - i, qreg_size - i]
        q_list += [2 * qreg_size + 1]

        natrual_gate | self(q_list)
