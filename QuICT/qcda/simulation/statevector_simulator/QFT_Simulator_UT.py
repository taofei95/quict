import numpy as np

from time import time

from QuICT.core import *
from QuICT.qcda.simulation.statevector_simulator.QFT_simulator import *

import qiskit
from qiskit.providers.aer import AerSimulator
from QuICT.algorithm import Amplitude


def time_count(qubits):
    circ = qiskit.QuantumCircuit(qubits)
    for i in range(qubits):
        circ.h(i)
        for j in range(i + 1, qubits):
            params = 2 * np.pi / (1 << j - i + 1)
            circ.crz(params, j, i)
    backend = AerSimulator(method='statevector', precision='single', device='CPU')
    start = time()
    qiskit.execute(circ, backend=backend, shots=1, optimization_level=0).result()
    end = time()
    return end - start

if __name__ == '__main__':
    f = open('test_out.csv', 'a')

    circuit_pre = Circuit(5)
    QFT.build_gate(5) | circuit_pre
    simulator_pre = QFTSimulator(circuit_pre)
    state = simulator_pre.run()

    qubits_range = range(23, 28)
    print("+++++++++++++++++++++++++++++++++++++++++++", file=f)
    for qubits in qubits_range:
        print(qubits)
        circuit = Circuit(qubits)
        QFT.build_gate(qubits) | circuit
        # state_exp = Amplitude.run(circuit)
        nst = time()
        simulator = QFTSimulator(circuit)
        state = simulator.run()
        net = time()
        # print(np.allclose(state_exp, state))
        print(f'{qubits}, {net - nst}, {time_count(qubits)}', file=f)
    
    f.close()
