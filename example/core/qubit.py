from QuICT.core import Qubit, Qureg


def build_qubit():
    # create a qubit
    q = Qubit(
        fidelity = 1.0,
        T1 = 0.0,
        T2 = 0.0
    )

    # Set fidelity and T1, T2 coherence time
    q.fidelity = 0.5
    q.T1 = 33.712
    q.T2 = 2.128

    print(q)


def build_qreg():
    qubit_number = 5
    qubit_list = [Qubit() for _ in range(5)]

    qreg1 = Qureg(qubit_number)
    qreg2 = Qureg(qubit_list)
    assert len(qreg1) == len(qreg2)

    # Set Fidelity/T1/T2 for Qureg
    qreg1.set_fidelity([0.5] * 5)
    qreg1.set_t1_time([30.1] * 5)
    qreg1.set_t2_time([2.3] * 5)
    print(qreg1[3])

    # Set Coupling Strength for Qureg with linearly topology
    cs = [(0, 1, 0.9), (1, 2, 0.91), (2, 3, 0.8), (3, 4, 0.6)]
    qreg1.set_coupling_strength(cs)
    print(qreg1.coupling_strength)


if __name__ == "__main__":
    build_qubit()
    build_qreg()
