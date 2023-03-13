from QuICT.core import Circuit
from QuICT.core.gate import *


# Build quantum circuit
circuit = Circuit(3)

X        | circuit(0)
H        | circuit(1)
H        | circuit(2)
CX       | circuit([1, 0])
T_dagger | circuit(0)
CX       | circuit([2, 0])
T        | circuit(0)
CX       | circuit([1, 0])
T_dagger | circuit(0)
CX       | circuit([2, 0])
T        | circuit(0)
T_dagger | circuit(1)
H        | circuit(0)
CX       | circuit([2, 1])
T_dagger | circuit(1)
CX       | circuit([2, 1])
T        | circuit(2)
S        | circuit(1)
H        | circuit(2)
H        | circuit(1)
X        | circuit(2)
X        | circuit(1)
H        | circuit(1)
CX       | circuit([2, 1])
H        | circuit(1)
X        | circuit(2)
X        | circuit(1)
H        | circuit(2)
H        | circuit(1)

circuit.draw(filename="ccx_dec")
