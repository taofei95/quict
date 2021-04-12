from QuICT.core import Circuit, CX, CCX, Swap, X, QFT, IQFT, CRz, Measure
from QuICT.core import GateBuilder, GATE_ID
from QuICT.qcda.synthesis import BEAAdder

def Set(qreg, N):
    """
    Set the qreg as N, using X gates on specific qubits
    """
    n = len(qreg)
    for i in range(n):
        if N % 2 == 1:
            X | qreg[n-1-i]
        N = N//2

def TestDraperAdder():
    n = 3
    for a in range(0, 4):
        for b in range(0, 5):
            circuit = Circuit(n * 2)
            qreg_a = circuit([i for i in range(n)])
            qreg_b = circuit([i for i in range(n, n * 2)])
            Set(qreg_a, a)
            Set(qreg_b, b)
            BEAAdder(3) | circuit
            Measure | circuit
            circuit.exec()
            # aa = int(qreg_a)
            bb = int(qreg_b)
            print("{0}+{1}={2}".format(str(a), str(b), str(bb)))

if __name__ == "__main__":
    testlist = ["DraperAdder"]
    for testname in testlist:
        locals()["Test" + testname]()