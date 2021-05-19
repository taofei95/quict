import numpy as np

from QuICT.core import Circuit, CompositeGate, CX, CCX, X, QFT, IQFT, CRz, Rz
from ..._synthesis import Synthesis


def DraperAdder(a, b):
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


def FourierAdderWired(a, phib):
    """ store Φ(a + b) in phib, but a is wired

    (phib) -> (phib'=Φ(a + b))

    Args:
        a(int):      unsigned integer, least n bits used
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


def FourierReverseAdderWired(a, phib):
    """ store Φ(b - a) or Φ(b - a + 2**(n+1)) in phib, but a is wired

    (phib) -> (phib')

    Args:
        a(int):      unsigned integer, least n bits used
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


def FourierAdderWiredCC(a, phib, c, dualControlled):
    """ FourierAdderWired with 2 control bits

    (phib,c) -> (phib'=Φ(a + b),c)

    Args:
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


def FourierReverseAdderWiredCC(a, phib, c, dualControlled):
    """ FourierReverseAdderWired with 2 control bits

    (phib,c) -> (phib',c)

    Args:
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


def FourierAdderModCC(a, N, phib, c, low, dualControlled=True):
    """ use FourierAdderWired/FourierAdderWiredCC to calculate (a+b)%N in Fourier space

    (phib=Φ(b),c,low) -> (phib'=Φ((a+b)%N),c,low)


    Args:
        a(int):      least n bits used as unsigned
        N(int):      least n bits used as unsigned
        phib(Qureg): the qureg stores b,        length is n+1,
        c(Qureg):    the control qubits,    length is 2 or 1, see dualControlled
        low(Qureg):  the clean ancillary qubit, length is 1,
        dualControlled(bool): if True, c[0] will be used; else c[0:1] will be used

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    FourierAdderWiredCC(a, phib, c, dualControlled=dualControlled)
    FourierReverseAdderWired(N, phib)
    IQFT(len(phib)) | phib
    CX | (phib[0], low)
    QFT(len(phib))  | phib
    FourierAdderWiredCC(N, phib, low, dualControlled=False)
    FourierReverseAdderWiredCC(a, phib, c, dualControlled=dualControlled)
    IQFT(len(phib)) | phib
    X | phib[0]
    CX | (phib[0], low)
    X | phib[0]
    QFT(len(phib))  | phib
    FourierAdderWiredCC(a, phib, c, dualControlled=dualControlled)


def FourierAdderMod(a, N, phib, low):
    """ use FourierAdderWired/FourierAdderWiredCC to calculate (a+b)%N in Fourier space. no control bits.

    (phib=Φ(b),low) -> (phib'=Φ((a+b)%N),low)


    Args:
        a(int):      least n bits used as unsigned
        N(int):      least n bits used as unsigned
        phib(Qureg): the qureg stores b,        length is n+1,
        low(Qureg):  the clean ancillary qubit, length is 1,

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    FourierAdderWired(a, phib)
    FourierReverseAdderWired(N, phib)
    IQFT(len(phib)) | phib
    CX | (phib[0], low)
    QFT(len(phib))  | phib
    FourierAdderWiredCC(N, phib, low, dualControlled=False)
    FourierReverseAdderWired(a, phib)
    IQFT(len(phib))  | phib
    X | phib[0]
    CX | (phib[0], low)
    X | phib[0]
    QFT(len(phib))  | phib
    FourierAdderWired(a, phib)


def FourierMultModC(a, N, x, phib, c, low):
    """ use FourierAdderModCC to calculate (b+ax)%N in Fourier space

    (phib=Φ(b),x,c,low) -> (phib'=Φ((b+ax)%N),x,c,low)


    Args:
        a(int):      least n bits used as unsigned
        N(int):      least n bits used as unsigned
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
        FourierAdderModCC(p * a % N, N, phib, (c, x[i]), low)  # p * a % N
        p = p * 2


def FourierMultMod(a, N, x, phib, low):
    """ use FourierAdderMod to calculate (b+ax)%N in Fourier space

    (phib=Φ(b),x,low) -> (phib'=Φ((b+ax)%N),x,low)


    Args:
        a(int):      least n bits used as unsigned
        N(int):      least n bits used as unsigned
        x(Qureg):    the qureg stores x,        length is n,
        phib(Qureg): the qureg stores b,        length is n+1,
        low(Qureg):  the clean ancillary qubit, length is 1,

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """

    n = len(phib) - 1
    p = 1
    for i in range(n - 1, -1, -1):
        FourierAdderModCC(p * a % N, N, phib, x[i], low, dualControlled=False)  # p * a % N
        p = p * 2


def MultModC(a, N, x, b, c, low):
    QFT(len(b))  | b
    FourierMultModC(a, N, x, b, c, low)
    IQFT(len(b)) | b


def ExGCD(a, b, coff):
    if b == 0:
        coff[0] = 1
        coff[1] = 0
        return a
    r = ExGCD(b, a % b, coff)
    t = coff[0]
    coff[0] = coff[1]
    coff[1] = t - a // b * coff[1]
    return r


def InverseMod(a, N):
    coff = [0, 0]
    r = ExGCD(a, N, coff)
    if r != 1:
        return None
    else:
        return coff[0] % N


def BEAAdderDecomposition(n):
    """ a circuit calculate a+b, a and b are gotten from some qubits.
    
    (a,b) -> (a,b'=a+b)

    Quregs:
        a: the qureg stores a, length is n,
        b: the qureg stores b, length is n,

    """
    circuit = Circuit(n * 2)
    qreg_a = circuit([i for i in range(n)])
    qreg_b = circuit([i for i in range(n, n * 2)])
    DraperAdder(qreg_a, qreg_b)
    return CompositeGate(circuit.gates)


BEAAdder = Synthesis(BEAAdderDecomposition)


def BEAAdderWiredDecomposition(n, a):
    """ a circuit calculate a+b, a is wired, and b are gotten from some qubits.
    
    (b) -> (b'=a+b)

    Quregs:
        b: the qureg stores b, length is n+1,

    """
    circuit = Circuit(n + 1)
    qreg_b = circuit([i for i in range(n + 1)])
    QFT(len(qreg_b)) | qreg_b
    FourierAdderWired(a, qreg_b)
    IQFT(len(qreg_b)) | qreg_b
    return CompositeGate(circuit.gates)


BEAAdderWired = Synthesis(BEAAdderWiredDecomposition)


def BEAReverseAdderWiredDecomposition(n, a):
    """ 
    (b) -> (b'=b-a or b-a+2**(n+1))

    Quregs:
        b: the qureg stores b, length is n+1,

    """
    circuit = Circuit(n + 1)
    qreg_b = circuit([i for i in range(n + 1)])
    QFT(len(qreg_b))  | qreg_b
    FourierReverseAdderWired(a, qreg_b)
    IQFT(len(qreg_b)) | qreg_b
    return CompositeGate(circuit.gates)


BEAReverseAdderWired = Synthesis(BEAReverseAdderWiredDecomposition)


def BEAAdderWiredCCDecomposition(n, a):
    """ 
    (b) -> (b'=a+b)

    Quregs:
        b: the qureg stores b, length is n,
        c: the control bits,   length is 2
    """
    circuit = Circuit(n + 3)
    qreg_b = circuit([i for i in range(n + 1)])
    qreg_c = circuit([i for i in range(n + 1, n + 3)])
    QFT(len(qreg_b))  | qreg_b
    FourierAdderWiredCC(a, qreg_b, qreg_c, dualControlled=True)
    IQFT(len(qreg_b)) | qreg_b
    return CompositeGate(circuit.gates)


BEAAdderWiredCC = Synthesis(BEAAdderWiredCCDecomposition)


def BEAReverseAdderWiredCCDecomposition(n, a):
    """ 
    (b,c) -> (b'=b-a,c)

    Quregs:
        b: the qureg stores b, length is n+1,
        c: the control bits,   length is 2
    """
    circuit = Circuit(n + 3)
    qreg_b = circuit([i for i in range(n + 1)])
    qreg_c = circuit([i for i in range(n + 1, n + 3)])
    QFT(len(qreg_b))  | qreg_b
    FourierReverseAdderWiredCC(a, qreg_b, qreg_c, dualControlled=True)
    IQFT(len(qreg_b)) | qreg_b
    return CompositeGate(circuit.gates)


BEAReverseAdderWiredCC = Synthesis(BEAReverseAdderWiredCCDecomposition)


def BEAAdderModCCDecomposition(n, a, N):
    """ use FourierAdderWired/FourierAdderWiredCC to calculate (a+b)%N in Fourier space

    (phib=Φ(b),c,low) -> (phib'=Φ((a+b)%N),c,low)

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
    FourierAdderModCC(a, N, qreg_b, qreg_c, qreg_low)
    IQFT(len(qreg_b)) | qreg_b
    return CompositeGate(circuit.gates)


BEAAdderModCC = Synthesis(BEAAdderModCCDecomposition)


def BEAAdderModDecomposition(n, a, N):
    """ use FourierAdderWired/FourierAdderWiredCC to calculate (a+b)%N in Fourier space. No cotrol bits

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
    FourierAdderMod(a, N, qreg_b, qreg_low)
    IQFT(len(qreg_b)) | qreg_b
    return CompositeGate(circuit.gates)


BEAAdderMod = Synthesis(BEAAdderModDecomposition)


def BEAMultModCDecomposition(n, a, N):
    """ use FourierAdderModCC to calculate (b+ax)%N in Fourier space

    (phib=Φ(b),x,c,low) -> (phib'=Φ((b+ax)%N),x,c,low)


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
    FourierMultModC(a, N, qreg_x, qreg_b, qreg_c, qreg_low)
    IQFT(len(qreg_b)) | qreg_b
    return CompositeGate(circuit.gates)


BEAMulModC = Synthesis(BEAMultModCDecomposition)


def BEAMultModDecomposition(n, a, N):
    """ use FourierAdderMod to calculate (b+ax)%N in Fourier space. No control bits

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
    FourierMultMod(a, N, qreg_x, qreg_b, qreg_low)
    IQFT(len(qreg_b)) | qreg_b
    return CompositeGate(circuit.gates)


BEAMulMod = Synthesis(BEAMultModDecomposition)


def BEACUaDecomposition(n, a, N):
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
    a_inv = InverseMod(a, N)

    circuit = Circuit(2 * n + 3)
    qreg_b = circuit([i for i in range(n + 1)])
    qreg_x = circuit([i for i in range(n + 1, 2 * n + 1)])
    qreg_c = circuit(2 * n + 1)
    qreg_low = circuit(2 * n + 2)

    MultModC(a, N, qreg_x, qreg_b, qreg_c, qreg_low)
    idx_start = 0
    idx_end = len(circuit.gates)
    for i in range(n):  # n bits swapped, b[0] always 0
        # controlledSwap | (c,x[i],b[i+1])
        CX | (qreg_b[i + 1], qreg_x[i])
        CCX | (qreg_c, qreg_x[i], qreg_b[i + 1])
        CX | (qreg_b[i + 1], qreg_x[i])
    # ReverseMultModC(a_inv,N,x,b,c,low)
    for index in range(idx_end - 1, idx_start - 1, -1):
        circuit.append(circuit.gates[index].inverse())
    return CompositeGate(circuit.gates)


BEACUa = Synthesis(BEACUaDecomposition)
