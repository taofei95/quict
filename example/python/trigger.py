from QuICT.core.circuit.circuit_extend import Trigger
from QuICT.core import Circuit
from QuICT.core.gate import *

import QuICT

print(QuICT.__file__)

cgate0 = CompositeGate()
X | cgate0(2)
cgate1 = CompositeGate()
X | cgate1(2)
X | cgate1(2)

gates = [cgate0, cgate1]

trigger = Trigger(1, gates)  # notice that indices of gates accord to circuit

c = Circuit(3)
H | c([0])
trigger | c(
    [0]
)  # trigger measure its target and then switch on the result to construct gates

from QuICT.simulation.gpu_simulator import ConstantStateVectorSimulator

sim = ConstantStateVectorSimulator()
amp = sim.run(c)
