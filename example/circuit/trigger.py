from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.operator import Trigger

from QuICT.simulation.state_vector import CircuitSimulator


cgate0 = CompositeGate()
X | cgate0(2)
cgate1 = CompositeGate()
X | cgate1(2)
X | cgate1(2)

gates = [cgate0, cgate1]

trigger = Trigger(1, gates)  # notice that indices of gates accord to circuit

c = Circuit(3)
H | c([0])
trigger | c([0])  # trigger measure its target and then switch on the result to construct gates


sim = CircuitSimulator()
amp = sim.run(c)

print(amp)
