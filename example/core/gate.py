from QuICT.core.gate import CompositeGate, H, CX, gate_builder, GateType, QFT, Rx
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
    H | cgate(0)
    with cgate:     # Build CompositeGate through Context
        CX & [0, 1]
        CX & [1, 2]

    # Add another compositeGate into it
    qft_gate = QFT(3)
    qft_gate | cgate

    cgate.draw("command")

    # Adjust CompositeGate
    lgate = cgate.pop()     # Pop the last CompositeGate
    cgate.adjust(2, [0, 2])
    cgate.draw("command")


def build_MCT_gate():
    # Build 5 qubits MCT gate with one ancillary qubit
    mct = MultiControlToffoli(aux_usage='no_aux')
    mct1_gate = mct(3)
    mct1_gate.draw(method='command', flatten=True)


if __name__ == "__main__":
    build_CompositeGate()
