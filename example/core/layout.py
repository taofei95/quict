from QuICT.core import Layout


def build_layout():
    # Build a linearly layout with 5 qubits
    layout = Layout(qubit_number=5, name="linearly")
    layout.add_edge(0, 1, directional=False, error_rate=1.0)
    layout.add_edge(1, 2, directional=False, error_rate=1.0)
    layout.add_edge(2, 3, directional=False, error_rate=1.0)
    layout.add_edge(3, 4, directional=False, error_rate=1.0)
    print(layout)

if __name__ == "__main__":
    build_layout()