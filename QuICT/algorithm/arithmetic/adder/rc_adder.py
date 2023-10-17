from numpy import pi

from QuICT.core.gate import CompositeGate, Ry, CU3
from QuICT.algorithm.qft import ry_QFT, ry_IQFT


class RCFourierAdderWired(CompositeGate):
    r"""
        A wired in-place adder in fourier basis. One of the addend is given classically and will be
        written into the adder when contructing the gate. For a n-qubit binary encoded addends `a`
        and a classically given integer `X`, this adder calculates the result and store it in place:

        $$
            \vert{a}\rangle_n \to \vert{X+a \mod 2^n}\rangle_n
        $$

        Applying this adder with `X = 5` on a 4-qubit sized register looks like:

                    ┌──────────┐┌──────────┐┌──────────┐
            q_0: |0>┤0         ├┤ ry(5π/8) ├┤0         ├
                    │          │├──────────┤│          │
            q_1: |0>┤1         ├┤ ry(5π/4) ├┤1         ├
                    │  cg_yQFT │├──────────┤│  cg_IQFT │
            q_2: |0>┤2         ├┤ ry(5π/2) ├┤2         ├
                    │          │└┬────────┬┘│          │
            q_3: |0>┤3         ├─┤ ry(5π) ├─┤3         ├
                    └──────────┘ └────────┘ └──────────┘

        Examples:
            >>> from QuICT.core import Circuit
            >>> from QuICT.algorithm.arithmetic import RCFourierAdderWired
            >>>
            >>> X = 5
            >>> circuit = Circuit(4)
            >>> RCFourierAdderWired(4, addend=X) | circuit

        Note:
            The quantum fourier transform used in this adder is ryQFT instead of regular QFT.

        References:
            [1]: "High Performance Quantum Modular Multipliers" by Rich Rines, Isaac Chuang
            <https://arxiv.org/abs/1801.01081>
    """

    def __init__(
        self,
        qreg_size: int,
        addend: int,
        controlled: bool = False,
        in_fourier: bool = False,
        out_fourier: bool = False,
        name: str = "RCAdder"
    ):
        """
            Construct a wired classical-quantum adder in fourier basis.

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

            Raises:
                GateParametersAssignedError: If `qreg_size` is smaller than 2.
        """

        self._addend = addend
        self._controlled = controlled

        super().__init__(name)

        if controlled:
            if not in_fourier:
                ry_QFT(qreg_size) | self(range(1, qreg_size + 1))
            self._build_ctl_phi_adder(qreg_size, addend)
            if not out_fourier:
                ry_IQFT(qreg_size) | self(range(1, qreg_size + 1))
        else:
            if not in_fourier:
                ry_QFT(qreg_size) | self
            self._build_phi_adder(qreg_size, addend)
            if not out_fourier:
                ry_IQFT(qreg_size) | self

    @property
    def is_controlled(self):
        """ A bool value indicating if the adder is a controlled gate. """
        return self._controlled

    @property
    def addend(self):
        """ The classical addend in use for contructing the adder gate. """
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
