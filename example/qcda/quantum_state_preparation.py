import numpy as np

from QuICT.qcda.synthesis import QuantumStatePreparation


def qsp_example():
    random_sv = np.random.random(1 << 4).astype(np.complex128)
    qsp = QuantumStatePreparation(method="unitary_decomposition", keep_phase=True)

    qsp_circuit = qsp.execute(random_sv)
    qsp_circuit.draw(method="command", flatten=True)


if __name__ == "__main__":
    qsp_example()
