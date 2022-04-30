from QuICT.simulation.cpu_simulator import CircuitSimulator
from .benchmarks import Benchmarks


def test_qft():
    for circ in Benchmarks.qft("small"):
        simulator = CircuitSimulator()
        _ = simulator.run(circ)


def test_single():
    for circ in Benchmarks.single_bit("small"):
        simulator = CircuitSimulator()
        _ = simulator.run(circ)


def test_qasm_export():
    for circ in Benchmarks.single_bit("small"):
        print(circ.qasm())
