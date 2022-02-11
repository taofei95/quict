Classical Simulator
===================
How to validate the correctness of the quantum algorithm is one of the most important part in the Quantum Computation.
Using classical machine to simulate the quantum circuit is a way to validate the quantum algorithm. In QuICT, the Simulator
is used to simulate the qubits' state during runing in the quantum circuit.

## graph here (simulator support status [cpu, gpu])

The simulator will return a data structure which stores the informations about the simulation of quantum circuit
    - device: the hardware
    - backend: the mode of simulator
    - shots: the repeat times of simulation
    - options: the parameters for simulator
    - time: spending times
    - counts: the dict of the measure results for each simulation

.. toctree::
   :maxdepth: 1

   modes.rst
