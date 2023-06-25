import os

from QuICT.core import Circuit, Layout
from QuICT.core.utils import GateType
from QuICT.qcda.mapping import SABREMapping


if __name__ == '__main__':
    layout_path = os.path.join(os.path.dirname(__file__), "../layout/ibmqx2_layout.json")
    layout = Layout.load_file(layout_path)

    circuit = Circuit(5)
    circuit.random_append(50, typelist=[GateType.cx])
    circuit.draw(filename='before_sabremapping')

    sabre = SABREMapping(layout)
    circuit_map = sabre.execute(circuit)
    circuit_map.draw(filename='after_sabremapping')

    circuit_initial_map = sabre.execute_initialMapping(circuit)
    print(circuit_initial_map)
