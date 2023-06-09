from QuICT.core.gate import CompositeGate, H, CX, gate_builder, GateType, QFT
from QuICT.core.gate import MultiControlToffoli


def build_BasicGate():
    my_gate_cx = CX & [1, 5]   # Get a CX Gate with qubit indexes 1, 5.
    print(my_gate_cx)

    my_gate_cu1 = gate_builder(
        gate_type=GateType.cu1,
        precision="double",
        params=[1 / 2],
        random_params=False
    )
    print(my_gate_cu1)


def build_CompositeGate():
    cgate = CompositeGate()
    with cgate:
        H & 0
        CX & [0, 1]
        CX & [1, 2]

    CX | cgate([2, 3])
    print(cgate.qasm())


def build_QFT():
    # Build 5-qubits QFT Gates
    n = 5
    qft_gate = QFT(n)
    qft_gate.draw(method="command")


def build_MCT_gate():
    # Build 5 qubits MCT gate with one ancillary qubit
    mct = MultiControlToffoli(aux_usage='one_clean_aux')
    mct1_gate = mct(5)
    mct1_gate.draw(method='command', flatten=True)


if __name__ == "__main__":
    build_MCT_gate()
