from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.noise import BitflipError
from QuICT.core.operator import Trigger, NoiseGate, CheckPoint

from QuICT.simulation.state_vector import StateVectorSimulator


def build_trigger():
    cgate0 = CompositeGate()
    X | cgate0(2)
    cgate1 = CompositeGate()
    X | cgate1(2)
    X | cgate1(2)

    gates = [
        cgate0,     # related gates if measured 0
        cgate1      # related gates if measured 1
    ]

    trigger = Trigger(1, gates)  # notice that indices of gates accord to circuit

    c = Circuit(3)
    H | c([0])
    trigger | c([0])  # trigger measure its target and then switch on the result to construct gates

    sim = StateVectorSimulator()
    amp = sim.run(c)
    print(amp)


def build_noisegate():
    error = BitflipError(0.1)
    gate = gate_builder(GateType.h)
    ng = NoiseGate(gate, error)
    print(ng.noise_matrix)


def build_checkpoint():
    # Normally, work with Trigger for more flexible
    cp = CheckPoint()
    cir = Circuit(4)
    cir.random_append(10)
    cp | cir
    cir.random_append(10)

    cpc = cp.get_child()
    cgate = CompositeGate("target")
    H | cgate(0)
    cpc | cgate

    cgate | cir
    print(cir.gates[10].name)


if __name__ == "__main__":
    build_checkpoint()
