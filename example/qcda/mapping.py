import os

from QuICT.core import Circuit, Layout
from QuICT.core.utils import GateType
from QuICT.qcda.mapping import MCTSMapping, SABREMapping


def mcts_mapping(circuit, layout):
    mcts = MCTSMapping(layout)
    circuit_map = mcts.execute(circuit)
    print("The result from MCTS Mapping")
    circuit_map.draw(method="command", flatten=True)


def sabre_mapping(circuit, layout):
    sabre = SABREMapping(layout)
    circuit_map = sabre.execute(circuit)
    print("The result from SABRE Mapping")
    circuit_map.draw(method="command", flatten=True)


def initial_mapping(circuit, layout):
    sabre = SABREMapping(layout)
    initial_map = sabre.execute_initialMapping(circuit)
    print(initial_map)


if __name__ == '__main__':
    layout_path = os.path.join(os.path.dirname(__file__), "../layout/ibmqx2_layout.json")
    layout = Layout.load_file(layout_path)
    circuit = Circuit(5)
    circuit.random_append(20, typelist=[GateType.cx])
    print("The original Quantum Circuit.")
    circuit.draw(method="command")

    mcts_mapping(circuit, layout)
    sabre_mapping(circuit, layout)
    initial_mapping(circuit, layout)
