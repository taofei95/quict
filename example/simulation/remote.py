from QuICT.core import Circuit
from QuICT.core.gate import GateType
from QuICT.simulation.remote.quafu_simulator import QuafuSimulator


def quafu_simulator():
    circuit = Circuit(5)
    circuit.random_append(100, [GateType.h, GateType.cx, GateType.rx, GateType.ry, GateType.rz])

    simu = QuafuSimulator(token="Personal Token Input Here.")
    res = simu.run(circuit=circuit)

    return res


if __name__ == "__main__":
    quafu_simulator()
