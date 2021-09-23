import numpy as np

from QuICT.core import Circuit, CompositeGate, CX, CCX, X, QFT, IQFT, CRz, Rz
from ..._synthesis import Synthesis


def draper_adder(a, b):
    """ store a + b in b

    (a,b) -> (a,b'=a+b)


    Args:
        a(Qureg): the qureg stores a, length is n
        b(Qureg): the qureg stores b, length is n

    """
    n = len(a)
    QFT(n) | b
    for i in range(n):
        p = 0
        for j in range(i, n):
            p += 1
            CRz(2 * np.pi / (1 << p)) | (a[j], b[i])
    IQFT(n) | b


def fourier_adder_wired(a, phib):
    """ store Φ(a + b) in phib, but a is wired

    (phib) -> (phib'=Φ(a + b))

    Args:
        a(int):      unsigned integer, low n bits used
        phib(Qureg): the qureg stores Φ(b), length is n+1

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    n = len(phib) - 1
    a = a & ~(1 << n)  # clear (n+1)-th bit to zero
    for i in range(n + 1):
        p = 0
        for j in range(i, n + 1):
            p += 1
            if a & (1 << (n - j)) != 0:
                Rz(2 * np.pi / (1 << p)) | phib[i]


def fourier_adder_wired_reversed(a, phib):
    """ store Φ(b - a) or Φ(b - a + 2**(n+1)) in phib, but a is wired

    (phib) -> (phib')

    Args:
        a(int):      unsigned integer, low n bits used
        phib(Qureg): the qureg stores Φ(b), length is n+1

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    n = len(phib) - 1
    a = a & ~(1 << n)  # clear (n+1)-th bit to zero
    for i in reversed(range(n + 1)):
        p = n + 1 - i
        for j in reversed(range(i, n + 1)):
            if a & (1 << (n - j)) != 0:
                Rz(- 2 * np.pi / (1 << p)) | phib[i]
            p -= 1


def cc_fourier_adder_wired(a, phib, c, dualControlled):
    """ fourier_adder_wired with 1 or 2 control bits

    (phib,c) -> (phib'=Φ(a + b),c)

    Args:
        a(int):      unsigned integer, low n bits used
        phib(Qureg): the qureg stores Φ(b), length is n
        c(Qureg):    the control qubits,    length is 2 or 1, see dualControlled
        dualControlled(bool): if True, c[0] will be used; else c[0:2] will be used

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    n = len(phib) - 1
    a = a & ~(1 << n)  # clear (n+1)-th bit to zero
    for i in range(n + 1):
        p = 0
        for j in range(i, n + 1):
            p += 1
            if a & (1 << (n - j)) != 0:
                # CCRz(2 * np.pi / (1 << p)) | (c[0],c[1],phib[i])
                # QCQI p128, Figure 4.8
                phase = 2 * np.pi / (1 << p)
                if dualControlled:
                    CRz(phase / 2) | (c[1], phib[i])
                    CX | (c[0], c[1])
                    CRz(- phase / 2) | (c[1], phib[i])
                    CX | (c[0], c[1])
                    CRz(phase / 2) | (c[0], phib[i])
                else:
                    CRz(phase) | (c[0], phib[i])


def cc_fourier_adder_wired_reversed(a, phib, c, dualControlled):
    """ fourier_adder_wired_reversed with 1 or 2 control bits

    (phib,c) -> (phib',c)

    Args:
        a(int):      unsigned integer, low n bits used
        phib(Qureg): the qureg stores Φ(b), length is n+1
        c(Qureg):    the control qubits,    length is 2 or 1, see dualControlled
        dualControlled(bool): default True. if True, c[0] will be used; else c[0:1] will be used

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    n = len(phib) - 1
    a = a & ~(1 << n)  # clear (n+1)-th bit to zero
    for i in reversed(range(n + 1)):
        p = n + 1 - i
        for j in reversed(range(i, n + 1)):
            if a & (1 << (n - j)) != 0:
                phase = 2 * np.pi / (1 << p)
                # CCRz(- 2 * np.pi / (1 << p)) | (c[0],c[1],phib[i])
                # QCQI p128, Figure 4.8
                if dualControlled:
                    CRz(- phase / 2) | (c[0], phib[i])
                    CX | (c[0], c[1])
                    CRz(phase / 2) | (c[1], phib[i])
                    CX | (c[0], c[1])
                    CRz(- phase / 2) | (c[1], phib[i])
                else:
                    CRz(- phase) | (c[0], phib[i])
            p -= 1

def cc_fourier_adder_mod(a, N, phib, c, low, dualControlled=True):
    """ use fourier_adder_wired/cc_fourier_adder_wired to calculate (a+b)%N in Fourier space

    (phib=Φ(b),c,low) -> (phib'=Φ((a+b)%N),c,low)


    Args:
        a(int):      low n bits used as unsigned
        N(int):      low n bits used as unsigned
        phib(Qureg): the qureg stores b,        length is n+1,
        c(Qureg):    the control qubits,    length is 2 or 1, see dualControlled
        low(Qureg):  the clean ancillary qubit, length is 1,
        dualControlled(bool): if True, c[0] will be used; else c[0:1] will be used

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    cc_fourier_adder_wired(a, phib, c, dualControlled=dualControlled)
    fourier_adder_wired_reversed(N, phib)
    IQFT(len(phib)) | phib
    CX | (phib[0], low)
    QFT(len(phib))  | phib
    cc_fourier_adder_wired(N, phib, low, dualControlled=False)
    cc_fourier_adder_wired_reversed(a, phib, c, dualControlled=dualControlled)
    IQFT(len(phib)) | phib
    X | phib[0]
    CX | (phib[0], low)
    X | phib[0]
    QFT(len(phib))  | phib
    cc_fourier_adder_wired(a, phib, c, dualControlled=dualControlled)


def fourier_adder_mod(a, N, phib, low):
    """ use fourier_adder_wired/cc_fourier_adder_wired to calculate (a+b)%N in Fourier space. no control bits.

    (phib=Φ(b),low) -> (phib'=Φ((a+b)%N),low)


    Args:
        a(int):      low n bits used as unsigned
        N(int):      low n bits used as unsigned
        phib(Qureg): the qureg stores b,        length is n+1,
        low(Qureg):  the clean ancillary qubit, length is 1,

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    fourier_adder_wired(a, phib)
    fourier_adder_wired_reversed(N, phib)
    IQFT(len(phib)) | phib
    CX | (phib[0], low)
    QFT(len(phib))  | phib
    cc_fourier_adder_wired(N, phib, low, dualControlled=False)
    fourier_adder_wired_reversed(a, phib)
    IQFT(len(phib))  | phib
    X | phib[0]
    CX | (phib[0], low)
    X | phib[0]
    QFT(len(phib))  | phib
    fourier_adder_wired(a, phib)


def c_fourier_mult_mod(a, N, x, phib, c, low):
    """ use cc_fourier_adder_mod to calculate (b+ax)%N in Fourier space

    (phib=Φ(b),x,c,low) -> (phib'=Φ((b+ax)%N),x,c,low)


    Args:
        a(int):      low n bits used as unsigned
        N(int):      low n bits used as unsigned
        x(Qureg):    the qureg stores x,        length is n,
        phib(Qureg): the qureg stores b,        length is n+1,
        c(Qureg):    the control qubits,        length is 1,
        low(Qureg):  the clean ancillary qubit, length is 1,

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """

    n = len(phib) - 1
    p = 1
    for i in range(n - 1, -1, -1):
        cc_fourier_adder_mod(p * a % N, N, phib, (c, x[i]), low)  # p * a % N
        p = p * 2


def fourier_mult_mod(a, N, x, phib, low):
    """ use fourier_adder_mod to calculate (b+ax)%N in Fourier space

    (phib=Φ(b),x,low) -> (phib'=Φ((b+ax)%N),x,low)


    Args:
        a(int):      low n bits used as unsigned
        N(int):      low n bits used as unsigned
        x(Qureg):    the qureg stores x,        length is n,
        phib(Qureg): the qureg stores b,        length is n+1,
        low(Qureg):  the clean ancillary qubit, length is 1,

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """

    n = len(phib) - 1
    p = 1
    for i in range(n - 1, -1, -1):
        cc_fourier_adder_mod(p * a % N, N, phib, x[i], low, dualControlled=False)  # p * a % N
        p = p * 2


def c_mult_mod(a, N, x, b, c, low):
    QFT(len(b))  | b
    c_fourier_mult_mod(a, N, x, b, c, low)
    IQFT(len(b)) | b


class BEAAdder(Synthesis):
    @classmethod
    def execute(cls, n):
        """ a circuit calculate a+b, a and b are gotten from some qubits.

        (a,b) -> (a,b'=a+b)

        Args:
            n(int): length of a and b
        """
        circuit = Circuit(n * 2)
        qreg_a = circuit([i for i in range(n)])
        qreg_b = circuit([i for i in range(n, n * 2)])
        draper_adder(qreg_a, qreg_b)
        return CompositeGate(circuit.gates)


class BEAAdderWired(Synthesis):
    @classmethod
    def execute(cls, n, a):
        """ a circuit calculate a+b, a is wired, and b are gotten from some qubits.

        (b) -> (b'=a+b)

        Args:
            n(int): length of a. b is in length n+1
            a(int): the operand to be added. low n bits used
        """
        circuit = Circuit(n + 1)
        qreg_b = circuit([i for i in range(n + 1)])
        QFT(len(qreg_b)) | qreg_b
        fourier_adder_wired(a, qreg_b)
        IQFT(len(qreg_b)) | qreg_b
        return CompositeGate(circuit.gates)


class BEAReverseAdderWired(Synthesis):
    @classmethod
    def execute(cls, n, a):
        """
        (b) -> (b'=b-a or b-a+2**(n+1))

        Args:
            n(int): length of a. b is in length n+1
            a(int): the operand to be subtracted. low n bits used
        """
        circuit = Circuit(n + 1)
        qreg_b = circuit([i for i in range(n + 1)])
        QFT(len(qreg_b))  | qreg_b
        fourier_adder_wired_reversed(a, qreg_b)
        IQFT(len(qreg_b)) | qreg_b
        return CompositeGate(circuit.gates)


class CCBEAAdderWired(Synthesis):
    @classmethod
    def execute(cls, n, a):
        """
        (b,c) -> (b'=a+b,c) if c=0b11 else (b'=b,c)

        Args:
            n(int): length of a. b is in length n+1
            a(int): the operand to be subtracted. low n bits used
        """
        circuit = Circuit(n + 3)
        qreg_b = circuit([i for i in range(n + 1)])
        qreg_c = circuit([i for i in range(n + 1, n + 3)])
        QFT(len(qreg_b))  | qreg_b
        cc_fourier_adder_wired(a, qreg_b, qreg_c, dualControlled=True)
        IQFT(len(qreg_b)) | qreg_b
        return CompositeGate(circuit.gates)


class CCBEAReverseAdderWired(Synthesis):
    @classmethod
    def execute(cls, n, a):
        """
        (b,c) -> (b'=b-a,c) if c=0b11 else (b'=b,c)

        Args:
            n(int): length of a. b is in length n+1
            a(int): the operand to be subtracted. low n bits used
        """
        circuit = Circuit(n + 3)
        qreg_b = circuit([i for i in range(n + 1)])
        qreg_c = circuit([i for i in range(n + 1, n + 3)])
        QFT(len(qreg_b))  | qreg_b
        cc_fourier_adder_wired_reversed(a, qreg_b, qreg_c, dualControlled=True)
        IQFT(len(qreg_b)) | qreg_b
        return CompositeGate(circuit.gates)


class CCBEAAdderMod(Synthesis):
    @classmethod
    def execute(cls, n, a, N):
        """ use fourier_adder_wired/cc_fourier_adder_wired to calculate (a+b)%N in Fourier space

        (phib=Φ(b),c,low) -> (phib'=Φ((a+b)%N),c,low) if c=0b11 else (phib'==Φ(b),c,low)

        Args:
            n(int):      bits len
            a(int):      least n bits used as unsigned
            N(int):      least n bits used as unsigned

        Quregs:
            phib(Qureg): the qureg stores b,        length is n+1,
            c(Qureg):    the control qubits,        length is 2,
            low(Qureg):  the clean ancillary qubit, length is 1,

        Circuit for Shor’s algorithm using 2n+3 qubits
        http://arxiv.org/abs/quant-ph/0205095v3
        """
        circuit = Circuit(n + 4)
        qreg_b = circuit([i for i in range(n + 1)])
        qreg_c = circuit([i for i in range(n + 1, n + 3)])
        qreg_low = circuit([i for i in range(n + 3, n + 4)])
        QFT(len(qreg_b))  | qreg_b
        cc_fourier_adder_mod(a, N, qreg_b, qreg_c, qreg_low)
        IQFT(len(qreg_b)) | qreg_b
        return CompositeGate(circuit.gates)


class BEAAdderMod(Synthesis):
    @classmethod
    def execute(cls, n, a, N):
        """ use fourier_adder_wired/cc_fourier_adder_wired to calculate (a+b)%N in Fourier space. No cotrol bits

        (phib=Φ(b),low) -> (phib'=Φ((a+b)%N),low)

        Args:
            n(int):      bits len
            a(int):      least n bits used as unsigned
            N(int):      least n bits used as unsigned

        Quregs:
            phib(Qureg): the qureg stores b,        length is n+1,
            low(Qureg):  the clean ancillary qubit, length is 1,

        Circuit for Shor’s algorithm using 2n+3 qubits
        http://arxiv.org/abs/quant-ph/0205095v3
        """
        circuit = Circuit(n + 2)
        qreg_b = circuit([i for i in range(n + 1)])
        qreg_low = circuit([i for i in range(n + 1, n + 2)])
        QFT(len(qreg_b))  | qreg_b
        fourier_adder_mod(a, N, qreg_b, qreg_low)
        IQFT(len(qreg_b)) | qreg_b
        return CompositeGate(circuit.gates)


class CBEAMulMod(Synthesis):
    @classmethod
    def execute(cls, n, a, N):
        """ use cc_fourier_adder_mod to calculate (b+ax)%N in Fourier space

        (phib=Φ(b),x,c,low) -> (phib'=Φ((b+ax)%N),x,c,low) if c=0b1 else (phib'=Φ(b),x,c,low)

        Args:
            n(int):      bits len
            a(int):      least n bits used as unsigned
            N(int):      least n bits used as unsigned

        Quregs:
            phib(Qureg): the qureg stores b,        length is n+1,
            x(Qureg):    the qureg stores x,        length is n,
            c(Qureg):    the control qubits,        length is 1,
            low(Qureg):  the clean ancillary qubit, length is 1,

        Circuit for Shor’s algorithm using 2n+3 qubits
        http://arxiv.org/abs/quant-ph/0205095v3
        """
        circuit = Circuit(2 * n + 3)
        qreg_b = circuit([i for i in range(n + 1)])
        qreg_x = circuit([i for i in range(n + 1, 2 * n + 1)])
        qreg_c = circuit(2 * n + 1)
        qreg_low = circuit(2 * n + 2)
        QFT(len(qreg_b)) | qreg_b
        c_fourier_mult_mod(a, N, qreg_x, qreg_b, qreg_c, qreg_low)
        IQFT(len(qreg_b)) | qreg_b
        return CompositeGate(circuit.gates)


class BEAMulMod(Synthesis):
    @classmethod
    def execute(cls, n, a, N):
        """ use fourier_adder_mod to calculate (b+ax)%N in Fourier space. No control bits

        (phib=Φ(b),x,low) -> (phib'=Φ((b+ax)%N),x,low)


        Args:
            n(int):      bits len
            a(int):      least n bits used as unsigned
            N(int):      least n bits used as unsigned

        Quregs:
            phib(Qureg): the qureg stores b,        length is n+1,
            x(Qureg):    the qureg stores x,        length is n,
            low(Qureg):  the clean ancillary qubit, length is 1,

        Circuit for Shor’s algorithm using 2n+3 qubits
        http://arxiv.org/abs/quant-ph/0205095v3
        """
        circuit = Circuit(2 * n + 2)
        qreg_b = circuit([i for i in range(n + 1)])
        qreg_x = circuit([i for i in range(n + 1, 2 * n + 1)])
        qreg_low = circuit(2 * n + 1)
        QFT(len(qreg_b)) | qreg_b
        fourier_mult_mod(a, N, qreg_x, qreg_b, qreg_low)
        IQFT(len(qreg_b)) | qreg_b
        return CompositeGate(circuit.gates)


class BEACUa(Synthesis):
    @classmethod
    def execute(cls, n, a, N):
        """ Controlled-U_a, ((a*x)MOD(N)) if c=1, else (x)

        (b=0,x,c,low) -> (b=0,x',c,low)

        Args:
            n(int):      bits len
            a(int):      least n bits used as unsigned
            N(int):      least n bits used as unsigned

        Quregs:
            phib(Qureg): the qureg stores b,        length is n+1,
            x(Qureg):    the qureg stores x,        length is n,
            c(Qureg):    the qureg stores c,        length is 1,
            low(Qureg):  the clean ancillary qubit, length is 1,

        Circuit for Shor’s algorithm using 2n+3 qubits
        http://arxiv.org/abs/quant-ph/0205095v3
        """
        # a_inv = InverseMod(a, N)

        circuit = Circuit(2 * n + 3)
        qreg_b = circuit([i for i in range(n + 1)])
        qreg_x = circuit([i for i in range(n + 1, 2 * n + 1)])
        qreg_c = circuit(2 * n + 1)
        qreg_low = circuit(2 * n + 2)

        c_mult_mod(a, N, qreg_x, qreg_b, qreg_c, qreg_low)
        idx_start = 0
        idx_end = len(circuit.gates)
        for i in range(n):  # n bits swapped, b[0] always 0
            # controlledSwap | (c,x[i],b[i+1])
            CX | (qreg_b[i + 1], qreg_x[i])
            CCX | (qreg_c, qreg_x[i], qreg_b[i + 1])
            CX | (qreg_b[i + 1], qreg_x[i])
        # Reversec_mult_mod(a_inv,N,x,b,c,low)
        for index in range(idx_end - 1, idx_start - 1, -1):
            circuit.append(circuit.gates[index].inverse())
        return CompositeGate(circuit.gates)
