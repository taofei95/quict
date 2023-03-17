from QuICT.core.gate import CompositeGate, H, CX


cgate = CompositeGate()
with cgate:
    H & 0
    CX & [0, 1]
    CX & [1, 2]

CX | cgate([2, 3])
print(cgate.qasm())
