from QuICT.algorithm import *
from QuICT.core import *
from QuICT.qcda.synthesis import HRSIncrementer, HRSCAdder, HRSCSub, HRSCCCompare, HRSCCAdderMod

def test1():
    circuit = Circuit(4)
    circuit.assign_initial_zeros()
    X | circuit(1)
    amplitude = Amplitude.run(circuit)
    print(amplitude)
    HRSIncrementer(2) | circuit
    amplitude = Amplitude.run(circuit)
    print(amplitude)

def test2():
    circuit = Circuit(5)
    circuit.assign_initial_zeros()
    X | circuit(0) # control = 1 !!!
    HRSCAdder(2, 2) | circuit
    amplitude = Amplitude.run(circuit)
    print(amplitude)

def test3():
    circuit = Circuit(5)
    circuit.assign_initial_zeros()
    X | circuit(0) # control = 1 !!!
    HRSCSub(2, 1) | circuit
    amplitude = Amplitude.run(circuit)
    print(amplitude)

def test4():
    circuit = Circuit(6)
    circuit.assign_initial_zeros()
    X | circuit(0) # control1 = 1
    X | circuit(1) # control2 = 1
    # b = 00

    HRSCCCompare(2, 1) | circuit # c = 1

    # expected result: 110001----49

    amplitude = Amplitude.run(circuit)
    print(amplitude)


def test5():
    circuit = Circuit(8)
    circuit.assign_initial_zeros()
    X | circuit(0) # control1 = 1
    X | circuit(1) # control2 = 1
    # b = 010 = 2
    X | circuit(3) 
    # len(g) needs to be larger than 1, so n needs to be larger than 2 !!!
    # indicator bit somtimes seems to change
    HRSCCAdderMod(3, 1, 2) | circuit # a = 1, N = 3, (b + a) % N = 1

    # expected result: 110100----52
    for i in range(8):
        Measure | circuit(i)
    circuit.exec()
    qreg = circuit([i for i in range(8)])
    y = int(qreg)
    print(y)
    # amplitude = Amplitude.run(circuit)
    # print(type(amplitude))
    # print(amplitude)

if __name__ == "__main__":
    s = "test"
    num = 5 # change the number to switch test function
    locals()[s + str(num)]()

