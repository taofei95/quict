from QuICT.core import Circuit, Layout
from QuICT.core.utils import GateType
from QuICT.qcda.mapping import MCTSMapping


if __name__ == '__main__':
    layout = Layout.load_file("../layout/ibmqx2.layout")

    circuit = Circuit(5)
    circuit.random_append(50, typelist=[GateType.cx])
    circuit.draw(filename='0.jpg')

    circuit_map = MCTSMapping.execute(circuit, layout)
    circuit_map.draw(filename='1.jpg')
