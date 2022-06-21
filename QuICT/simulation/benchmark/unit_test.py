from .benchmark_run import quict_sim, qiskit_sim


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
