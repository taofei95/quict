from QuICT.core import Circuit, CX, CCX, Swap, X, QFT, IQFT, CRz, Measure
from QuICT.core import GateBuilder, GATE_ID
from QuICT.qcda.synthesis.arithmetic.bea import *

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

def TestFourierAdderWired():
    n = 3
    for a in range(0, 8):
        for b in range(0, 8):
            circuit = Circuit(n+1)
            qreg_b = circuit([i for i in range(n+1)])
            Set(qreg_b, b)
            BEAAdderWired(3,a) | circuit
            Measure | circuit
            circuit.exec()
            # aa = int(qreg_a)
            bb = int(qreg_b)
            # print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
            assert bb==a+b

def TestFourierAdderWiredCC():
    n = 3
    for c in (0,1,2,3):
        if c!=3:
            print(format(c,"02b"),"disabled")
        else:
            print(format(c,"02b"),"enabled")
        for a in range(0, 8):
            for b in range(0, 8):
                circuit = Circuit(n+3)
                qreg_b = circuit([i for i in range(n+1)])
                qreg_c = circuit([i for i in range(n+1,n+3)])
                Set(qreg_b, b)
                Set(qreg_c, c)
                BEAAdderWiredCC(3,a) | circuit
                Measure | circuit
                circuit.exec()
                # aa = int(qreg_a)
                bb = int(qreg_b)
                #print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
                if c!=3:
                    assert bb==b
                else:
                    assert bb==a+b

def TestFourierReverseAdderWired():
    n = 3
    for a in range(0, 8):
        for b in range(0, 8):
            circuit = Circuit(n+1)
            qreg_b = circuit([i for i in range(n+1)])
            Set(qreg_b, b)
            BEAReverseAdderWired(3,a) | circuit
            Measure | circuit
            circuit.exec()
            # aa = int(qreg_a)
            bb = int(qreg_b)
            # print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
            if b-a>=0:
                assert bb==b-a
            else:
                assert bb==(1<<(n+1))+b-a

def TestFourierReverseAdderWiredCC():
    n = 3
    for c in (0,1,2,3):
        if c!=3:
            print(format(c,"02b"),"disabled")
        else:
            print(format(c,"02b"),"enabled")
        for a in range(0, 8):
            for b in range(0, 8):
                circuit = Circuit(n+3)
                qreg_b = circuit([i for i in range(n+1)])
                qreg_c = circuit([i for i in range(n+1,n+3)])
                Set(qreg_b, b)
                Set(qreg_c, c)
                BEAReverseAdderWiredCC(3,a) | circuit
                Measure | circuit
                circuit.exec()
                # aa = int(qreg_a)
                bb = int(qreg_b)
                #print("{0}+{1}={2}".format(str(a), str(b), str(bb)))
                if c!=3:
                    assert bb==b
                else:
                    if b-a>=0:
                        assert bb==b-a
                    else:
                        assert bb==(1<<(n+1))+b-a

def TestFourierAdderMod():
    n = 3
    for N in (2,3,5,6):
        print("N=="+str(N))
        _TestFourierAdderModSingle(n,N)
    
def _TestFourierAdderModSingle(n,N):
    for c in (3,):
        if c!=3:
            print(format(c,"02b"),"disabled")
        else:
            print(format(c,"02b"),"enabled")
        for a in range(0, N):
            for b in range(0, N):
                circuit = Circuit(n+4)
                qreg_b = circuit([i for i in range(n+1)])
                qreg_c = circuit([i for i in range(n+1,n+3)])
                Set(qreg_b, b)
                Set(qreg_c, c)
                BEAAdderMod(n,a,N) | circuit
                Measure | circuit
                circuit.exec()
                # aa = int(qreg_a)
                bb = int(qreg_b)
                low = int(circuit(n + 3))
                assert low == 0
                # print("({0}+{1}) % {3}={2}".format(str(a), str(b), str(bb),str(N)))
                if c!=3:
                    assert bb==b
                else:
                    assert bb==(a+b)%N

def TestBEAMultMod():
    n = 3
    for N in range(0, 8):
        for a in range(0, N):
            for x in range(0, N):
                circuit = Circuit(2*n+3)
                qreg_b  = circuit([i for i in range(n+1)])
                qreg_x  = circuit([i for i in range(n+1,2*n+1)])
                qreg_c = circuit(2*n+1)
                Set(qreg_c, 1)
                Set(qreg_b, 0)
                Set(qreg_x, x)
                BEAMultMod(n, a, N) | circuit
                Measure | circuit
                circuit.exec()
                bb = int(qreg_b)
                # print("0 + {0}*{1} mod {2}={3}".format(str(a), str(x), str(N), str(bb)))
                assert bb == (0 + a * x) % N

def ExGCD(a,b,coff):
    if b==0:
        coff[0]=1
        coff[1]=0
        return a
    r = ExGCD(b,a%b,coff)
    t       = coff[0]
    coff[0] = coff[1]
    coff[1] = t - a//b * coff[1]
    return r

def TestBEACUa():
    n = 3
    for c in (1,):
        if c==0:
            print("disabled")
        else:
            print("enabled")
        for N in range(0, 8):
            for a in range(0, N):
                coff = [0,0]
                r = ExGCD(a,N,coff)
                if r!=1:
                    continue
                for x in range(0, N):
                    circuit = Circuit(2*n+3)
                    qreg_b  = circuit([i for i in range(n+1)])
                    qreg_x  = circuit([i for i in range(n+1,2*n+1)])
                    qreg_c = circuit(2*n+1)
                    Set(qreg_c, c)
                    Set(qreg_b, 0)
                    Set(qreg_x, x)
                    BEACUa(n, a, N) | circuit
                    Measure | circuit
                    circuit.exec()
                    xx = int(qreg_x)
                    bb = int(qreg_b)
                    print("{0}*{1} mod {2}={3}".format(str(a), str(x), str(N), str(xx)))
                    if c==0:
                        assert xx == x
                    else:
                        assert xx == (a * x) % N
                    # assert bb == 0

if __name__ == "__main__":
    testlist = ["DraperAdder","FourierAdderWired","FourierAdderWiredCC","FourierReverseAdderWiredCC","FourierReverseAdderWired","FourierAdderMod","BEAMultMod",]
    newlist  = ["BEACUa",] 
    """
    for x in xlist:
        print("------TEST:"+x+"------")
        locals()["Test" + x]()
    """
    for testname in newlist:
        print("------TEST:"+testname+"------")
        locals()["Test" + testname]()