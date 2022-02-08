Circuit
=======
Analogous to the way a classical computer is built from an electrical circuit containing wires and logic gates,
a quantum computer is built from a quantum circuit containing wires and elementary quantum gates to carry around
and manipulate the quantum information. [Quantum Computation and Quantum Information]

In QuICT, the quantum circuit is the most important unit, it supports the Quantum Algorithm design and the simulator running.
The Circuit is a class in QuICT, and it uses the qubits and gates to represent the quantum circuit's wires and quantum gates.
In Circuit, it assigns the quantum gates with its wires, and it allows append random gates and the Supremace circuit. The Circuit
also provides the tools to draw the quantum circuit and translate itself into the OpenQASM file.

How to Build the Circuit in QuICT
---------------------------------
The Circuit provides the append and extend to add gates into the quantum circuit; meanwhile, we can using the operation or(|) to
add gate into the quantum circuit.

.. code-block:: python
    from QuICT.core import Circuit
    from QuICT.core.gate import *

    # Build a circuit with qubits 5
    circuit = Circuit(5)

    # add gates
    H | circuit                     # append H gate to all qubits
    S | circuit(3)                  # append S gate with target qubit 3
    U2(1, 0) | circuit(1)           # append U2 gate with target qubit 1
    mg = H & 3
    circuit.append(mg)              # append mg gate
    QFT.build_gate(5) | circuit     # append QFT composite gate

    # append random gates and Supremace circuit
    circuit.random_append(rand_size=10)
    circuit.supremacy_append(repeat=4, pattern="ABCDCDAB")

How to Visualize the Circuit in QuICT
-------------------------------------
There are two ways to visualize the quantum circuit in QuICT, one is translate the Circuit with the OpenQASM style, the other one is
draw the graph describe the quantum circuit. By the way, not all quantum gates in QuICT are supportted by OpenQASM.

.. code-block:: python
    # circuit from above
    # qasm
    circuit.qasm()

    # draw the circuit's graph
    circuit.draw(method='matp', filename="QuICT")

For further details, please read the examples in example/python or example/demo.
