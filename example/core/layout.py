import os

from QuICT.core import Layout


def build_layout():
    # Build a linearly layout with 5 qubits
    layout = Layout(qubit_number=5)
    layout.add_edge(0, 1, directional=False, error_rate=1.0)
    layout.add_edge(1, 2, directional=False, error_rate=1.0)
    layout.add_edge(2, 3, directional=False, error_rate=1.0)
    layout.add_edge(3, 4, directional=False, error_rate=1.0)
    print(layout)

    # Save layout to file
    layout.write_file()


def build_linear_or_grid_layout():
    # Build a linearly layout with 5 qubits
    layout = Layout(qubit_number=5, name="linearly")
    layout.linear_layout(directional=False, error_rate=[0.99] * 4)
    print(layout)

    # Build a grid layout with 3*3 qubits
    layout = Layout(qubit_number=9, name="grid")
    layout.grid_layout()
    print(layout)

    layout = Layout(qubit_number=8, name="grid2*4")
    layout.grid_layout(width=4)
    print(layout)


def load_layout():
    # From file
    layout_path = os.path.join(os.path.dirname(__file__), "../layout/ibmqx2_layout.json")
    layout = Layout.load_file(layout_path)
    print(layout)

    # From Json
    layout_json = layout.to_json()
    layout = Layout.from_json(layout_json)
    print(layout)


if __name__ == "__main__":
    build_layout()
