from .benchmark_run import quict_sim, qiskit_sim


def test_quict():
    quict_sim("small")
