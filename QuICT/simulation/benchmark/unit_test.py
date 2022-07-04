from .benchmark_run import quict_sim, qiskit_sim


def test_quict():
    quict_sim("small")
    quict_sim("medium")

def test_qiskit():
    qiskit_sim("small")
    qiskit_sim("medium")
