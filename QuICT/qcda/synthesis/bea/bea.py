
from numpy import log2, floor, gcd
import numpy as np
import sys
from .._synthesis import Synthesis
from QuICT.core import Circuit, CX, CCX, Swap, X, QFT, IQFT, CRz, Rz, Measure
from QuICT.core import GateBuilder, GATE_ID

def DraperAdder(a, b):
    """ store a + b in b

    (a,b) -> (a,b'=a+b)


    Args:
        a(Qureg): the qureg stores a, length is n
        b(Qureg): the qureg stores b, length is n

    """
    n = len(a)
    QFT | b
    for i in range(n):
        p = 0
        for j in range(i, n):
            p += 1
            CRz(2 * np.pi / (1 << p)) | (a[j], b[i])
    IQFT | b    

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
    a = a & ~(1<<n)         #clear (n+1)-th bit to zero
    for i in range(n+1):
        p = 0
        for j in range(i, n+1):
            p += 1
            if a & (1<<(n-j)) != 0:
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
    a = a & ~(1<<n)         #clear (n+1)-th bit to zero
    for i in reversed(range(n+1)):
        p = n+1-i
        for j in reversed(range(i, n+1)):
            if a & (1<<(n-j)) != 0:
                Rz(- 2 * np.pi / (1 << p)) | phib[i]
            p -= 1

def FourierAdderWiredCC(a, phib,c,dualControlled):
    """ FourierAdderWired with 2 control bits

    (phib,c) -> (phib'=Φ(a + b),c)

    Args:
        phib(Qureg): the qureg stores Φ(b), length is n
        c(Qureg):    the control qubits,    length is 2

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    n = len(phib) - 1
    a = a & ~(1<<n)         #clear (n+1)-th bit to zero
    for i in range(n+1):
        p = 0
        for j in range(i, n+1):
            p += 1
            if a & (1<<(n-j)) != 0:
                #CCRz(2 * np.pi / (1 << p)) | (c[0],c[1],phib[i])
                #QCQI p128, Figure 4.8
                phase = 2 * np.pi / (1 << p)
                if dualControlled:
                    CRz(phase / 2)  | (c[1],phib[i])
                    CX              | (c[0],c[1])
                    CRz(- phase / 2)| (c[1],phib[i])
                    CX              | (c[0],c[1])
                    CRz(phase / 2)  | (c[0],phib[i])
                else:
                    CRz(phase)      | (c[0],phib[i])

def FourierReverseAdderWiredCC(a, phib,c,dualControlled):
    """ FourierReverseAdderWired with 2 control bits

    (phib,c) -> (phib',c)

    Args:
        phib(Qureg): the qureg stores Φ(b), length is n+1
        c(Qureg):    the control qubits,    length is 2

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    n = len(phib) - 1
    a = a & ~(1<<n)         #clear (n+1)-th bit to zero
    for i in reversed(range(n+1)):
        p = n+1-i
        for j in reversed(range(i, n+1)):
            if a & (1<<(n-j)) != 0:
                phase = 2 * np.pi / (1 << p)
                #CCRz(- 2 * np.pi / (1 << p)) | (c[0],c[1],phib[i])
                #QCQI p128, Figure 4.8
                if dualControlled:
                    CRz(- phase / 2)| (c[0],phib[i])
                    CX              | (c[0],c[1])
                    CRz(phase / 2)  | (c[1],phib[i])
                    CX              | (c[0],c[1])
                    CRz(- phase / 2)| (c[1],phib[i])
                else:
                    CRz(- phase)    | (c[0],phib[i])
            p -= 1

def FourierAdderMod(a, N, phib, c, low):
    """ use FourierAdderWired/FourierAdderWiredCC to calculate (a+b)%N in Fourier space

    (phib=Φ(b),c,low) -> (phib'=Φ((a+b)%N),c,low)


    Args:
        a(int):      least n bits used as unsigned
        N(int):      least n bits used as unsigned
        phib(Qureg): the qureg stores b,        length is n+1,
        c(Qureg):    the control qubits,        length is 2,
        low(Qureg):  the clean ancillary qubit, length is 1,

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    FourierAdderWiredCC(a,phib,c,dualControlled=True)
    FourierReverseAdderWired(N,phib)
    IQFT | phib
    CX   | (phib[0],low)
    QFT  | phib
    FourierAdderWiredCC(N,phib,low,dualControlled=False)
    FourierReverseAdderWiredCC(a,phib,c,dualControlled=True)
    IQFT | phib
    X    | phib[0]
    CX   | (phib[0],low)
    X    | phib[0]
    QFT  | phib
    FourierAdderWiredCC(a,phib,c,dualControlled=True)

def FourierMultMod(a, N, x, phib, c, low):
    """ use FourierAdderMod to calculate (b+ax)%N in Fourier space

    (phib=Φ(b),x,c,low) -> (phib'=Φ((b+ax)%N),x,c,low)


    Args:
        a(int):      least n bits used as unsigned
        N(int):      least n bits used as unsigned
        x(Qureg):    the qureg stores x,        length is n,
        phib(Qureg): the qureg stores b,        length is n+1,
        c(Qureg):    the control qubits,        length is 2,
        low(Qureg):  the clean ancillary qubit, length is 1,

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """

    n = len(phib) - 1
    p = 1
    for i in range(n - 1, -1, -1):
        FourierAdderMod(p * a % N, N, phib, (c, x[i]), low) # p * a % N
        p = p * 2

class BEAModel(Synthesis):
    def __call__(self, *pargs):
        """ 
        Calling this empty class makes no effect
        """
        raise Exception('Calling this empty class makes no effect')

    def build_gate(self):
        """ 
        Empty class builds no gate
        """
        raise Exception('Empty class builds no gate')

BEA = BEAModel()

class BEAAdderModel(BEAModel):
    """ a circuit calculate a+b, a and b are gotten from some qubits.
    
    (a,b) -> (a,b'=a+b)

    Quregs:
        a: the qureg stores a, length is n,
        b: the qureg stores b, length is n,

    """
    def __call__(self,n):
        """ Overload the function __call__,
        Give parameters to the BEA.

        Args:
            n: the length of a and b.
        Returns:
            BEAAdderModel: the model filled by parameters.
        """

        self.pargs = [n]
        return self

    def build_gate(self):
        """ Overload the function build_gate.

        Returns:
            Circuit: the BEA circuit
        """
        n = self.pargs[0]

        circuit = Circuit(n * 2)
        qreg_a = circuit([i for i in range(n)])
        qreg_b = circuit([i for i in range(n, n * 2)])
        DraperAdder(qreg_a, qreg_b)
        return circuit

BEAAdder = BEAAdderModel()

class BEAAdderWiredModel(BEAModel):
    """ a circuit calculate a+b, a is wired, and b are gotten from some qubits.
    
    (b) -> (b'=a+b)

    Quregs:
        b: the qureg stores b, length is n+1,

    """
    def __call__(self,n,a):
        """ Overload the function __call__,
        Give parameters to the BEA.

        Args:
            n: the length of number b.
            a: 
        Returns:
            BEAAdderModel: the model filled by parameters.
        """

        self.pargs = [n,a]
        return self

    def build_gate(self):
        """ Overload the function build_gate.

        Returns:
            Circuit: the BEA circuit
        """
        n = self.pargs[0]
        a = self.pargs[1]

        circuit = Circuit(n+1)
        qreg_b = circuit([i for i in range(n+1)])
        QFT | qreg_b
        FourierAdderWired(a,qreg_b)
        IQFT | qreg_b
        return circuit

BEAAdderWired = BEAAdderWiredModel()

class BEAReverseAdderWiredModel(BEAModel):
    """ 
    (b) -> (b'=b-a or b-a+2**(n+1))

    Quregs:
        b: the qureg stores b, length is n+1,

    """
    def __call__(self,n,a):
        """ Overload the function __call__,
        Give parameters to the BEA.

        Args:
            n: the length of number b.
            a: 
        Returns:
            BEAAdderModel: the model filled by parameters.
        """

        self.pargs = [n,a]
        return self

    def build_gate(self):
        """ Overload the function build_gate.

        Returns:
            Circuit: the BEA circuit
        """
        n = self.pargs[0]
        a = self.pargs[1]

        circuit = Circuit(n+1)
        qreg_b = circuit([i for i in range(n+1)])
        QFT | qreg_b
        FourierReverseAdderWired(a,qreg_b)
        IQFT | qreg_b
        return circuit

BEAReverseAdderWired = BEAReverseAdderWiredModel()

class BEAAdderWiredCCModel(BEAModel):
    """ 
    (b) -> (b'=a+b)

    Quregs:
        b: the qureg stores b, length is n,
        c: the control bits,   length is 2
    """
    def __call__(self,n,a):
        """ Overload the function __call__,
        Give parameters to the BEA.

        Args:
            n: the length of b.
            a: 
        Returns:
            BEAAdderModel: the model filled by parameters.
        """

        self.pargs = [n,a]
        return self

    def build_gate(self):
        """ Overload the function build_gate.

        Returns:
            Circuit: the BEA circuit
        """
        n = self.pargs[0]
        a = self.pargs[1]

        circuit = Circuit(n+3)
        qreg_b = circuit([i for i in range(n+1)])
        qreg_c = circuit([i for i in range(n+1,n+3)])
        QFT | qreg_b
        FourierAdderWiredCC(a,qreg_b,qreg_c,dualControlled=True)
        IQFT | qreg_b
        return circuit

BEAAdderWiredCC = BEAAdderWiredCCModel()

class BEAReverseAdderWiredCCModel(BEAModel):
    """ 
    (b,c) -> (b'=b-a,c)

    Quregs:
        b: the qureg stores b, length is n+1,
        c: the control bits,   length is 2
    """
    def __call__(self,n,a):
        """ Overload the function __call__,
        Give parameters to the BEA.

        Args:
            n: the length of b.
            a: 
        Returns:
            BEAAdderModel: the model filled by parameters.
        """

        self.pargs = [n,a]
        return self

    def build_gate(self):
        """ Overload the function build_gate.

        Returns:
            Circuit: the BEA circuit
        """
        n = self.pargs[0]
        a = self.pargs[1]

        circuit = Circuit(n+3)
        qreg_b = circuit([i for i in range(n+1)])
        qreg_c = circuit([i for i in range(n+1,n+3)])
        QFT | qreg_b
        FourierReverseAdderWiredCC(a,qreg_b,qreg_c,dualControlled=True)
        IQFT | qreg_b
        return circuit

BEAReverseAdderWiredCC = BEAReverseAdderWiredCCModel()

class BEAAdderModModel(BEAModel):
    """ use FourierAdderWired/FourierAdderWiredCC to calculate (a+b)%N in Fourier space

    (phib=Φ(b),c,low) -> (phib'=Φ((a+b)%N),c,low)

    Args:
        a(int):      least n bits used as unsigned
        N(int):      least n bits used as unsigned
        phib(Qureg): the qureg stores b,        length is n+1,
        c(Qureg):    the control qubits,        length is 2,
        low(Qureg):  the clean ancillary qubit, length is 1,

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    def __call__(self,n,a,N):
        """ Overload the function __call__,
        Give parameters to the BEA.

        Args:
            n: the length of number a and b.
            a:
        Returns:
            BEAAdderModModel: the model filled by parameters.
        """
        self.pargs = [n,a,N]
        return self

    def build_gate(self):
        """ Overload the function build_gate.

        Returns:
            Circuit: the BEA circuit
        """
        n = self.pargs[0]
        a = self.pargs[1]
        N = self.pargs[2]

        circuit = Circuit(n + 4)
        qreg_b  = circuit([i for i in range(n+1)])
        qreg_c  = circuit([i for i in range(n+1,n+3)])
        qreg_low= circuit([i for i in range(n+3,n+4)])
        QFT | qreg_b
        FourierAdderMod(a,N,qreg_b,qreg_c,qreg_low)
        IQFT| qreg_b
        return circuit

BEAAdderMod = BEAAdderModModel()

class BEAMultModModel(BEAModel):
    """ use FourierAdderMod to calculate (b+ax)%N in Fourier space

    (phib=Φ(b),x,c,low) -> (phib'=Φ((b+ax)%N),x,c,low)


    Args:
        a(int):      least n bits used as unsigned
        N(int):      least n bits used as unsigned
        x(Qureg):    the qureg stores x,        length is n,
        phib(Qureg): the qureg stores b,        length is n+1,
        c(Qureg):    the control qubits,        length is 2,
        low(Qureg):  the clean ancillary qubit, length is 1,

    Circuit for Shor’s algorithm using 2n+3 qubits
    http://arxiv.org/abs/quant-ph/0205095v3
    """
    def __call__(self,n,a,N):
        """ Overload the function __call__,
        Give parameters to the BEA.

        Args:
            n: the length of number a and x.
            a:
            N: the modulus
        Returns:
            BEAMultModModel: the model filled by parameters.
        """
        self.pargs = [n,a,N]
        return self

    def build_gate(self):
        """ Overload the function build_gate.

        Returns:
            Circuit: the BEA circuit
        """
        n = self.pargs[0]
        a = self.pargs[1]
        N = self.pargs[2]

        circuit = Circuit(2 * n + 3)
        qreg_b  = circuit([i for i in range(n+1)])
        qreg_x  = circuit([i for i in range(n+1,2*n+1)])
        qreg_c = circuit(2 * n + 1)
        qreg_low= circuit(2 * n + 2)
        QFT | qreg_b
        FourierMultMod(a, N, qreg_x, qreg_b, qreg_c, qreg_low)
        IQFT| qreg_b
        return circuit

BEAMultMod = BEAMultModModel()