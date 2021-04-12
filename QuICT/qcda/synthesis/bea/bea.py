
from numpy import log2, floor, gcd
import numpy as np
import sys
from .._synthesis import Synthesis
from QuICT.core import Circuit, CX, CCX, Swap, X, QFT, IQFT, CRz, Measure
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

