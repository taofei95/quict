import os

from QuICT.core import Circuit, Layout
from QuICT.core.utils import GateType
from QuICT.qcda.mapping import MCTSMapping


if __name__ == '__main__':
    layout_path = os.path.join(os.path.dirname(__file__), "ibmqx2_layout.json")
    layout = Layout.load_file(layout_path)

    circuit = Circuit(5)
    circuit.random_append(50, typelist=[GateType.cx])
    circuit.draw(filename='before_mapping')

    mcts = MCTSMapping(layout)
    circuit_map = mcts.execute(circuit)
    circuit_map.draw(filename='after_mapping')
