State-Vector Simulator
======================
The state-vector simulator holds the qubits' states during running the quantum circuit. After running
through the given quantum circuit, it returns the final qubits' states.

Example
>>>>>>>


Unitary Simulator
======================
The unitary simulator is used to generate the unitary matrix of the given quantum circuit. Not like the other
classical simulator, the unitary simulator do not care about the qubits' states, it only returns the unitary matrix
of the quantum algorithm.

Example
>>>>>>>

Multi-GPU State-Vector Simulator
================================
During the incresment of the qubits, the required memory of simulation is exponential increasing. The Multi-GPU State-Vector simulator
is designed to use multi-gpus in one machine to simulate the running of the quantum circuit; therefore, the simulator can be faster and more extensive.

Example
>>>>>>>

Remote Simulator
================
Currently the QuICT supports to simulate with the simulator from other platform (Qiskit and QCompute).

Example
>>>>>>>