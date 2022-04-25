from .circuits import *
from QuICT.simulation.cpu_simulator import CircuitSimulator


def test_qft():
    for circ in qft_circuits():
        simulator = CircuitSimulator()
        amp = simulator.run(circ)
