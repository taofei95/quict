
from QuICT.core import Circuit
from QuICT.core.utils import GateType
from QuICT.qcda.optimization.commutative_optimization import CommutativeOptimization


# Be aware that too many types at the same time may not benefit to the test,
# unless the size of the random circuit is also large.
typelist = [
    GateType.rx, GateType.ry, GateType.rz, GateType.x,
    GateType.y, GateType.z, GateType.cx    
]

if __name__ == '__main__':
    circuit = Circuit(5)
    circuit.random_append(rand_size=100, typelist=typelist)
    circuit.draw()

    gates = CommutativeOptimization.execute(circuit)
    circuit_opt = Circuit(5)
    circuit_opt.extend(gates.gates)
    circuit_opt.draw()
