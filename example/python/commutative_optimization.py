
from QuICT.core import *
from QuICT.qcda.optimization.commutative_optimization import CommutativeOptimization

# Be aware that too many types at the same time may not benefit to the test,
# unless the size of the random circuit is also large.
typelist = [GATE_ID['Rx'], GATE_ID['Ry'], GATE_ID['Rz'],
            GATE_ID['X'], GATE_ID['Y'], GATE_ID['Z'], GATE_ID['CX']]

if __name__ == '__main__':
    circuit = Circuit(5)
    circuit.random_append(rand_size=100, typeList=typelist)
    circuit.draw()

    gates = CommutativeOptimization.execute(circuit)
    circuit_opt = Circuit(5)
    circuit_opt.set_exec_gates(gates)
    circuit_opt.draw()