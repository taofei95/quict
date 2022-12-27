# 核心
Circuit
=======
Analogous to the way a classical computer is built from an electrical circuit containing wires and logic gates,
a quantum computer is built from a quantum circuit containing wires and elementary quantum gates to carry around
and manipulate the quantum information.

In QuICT, the quantum circuit is the most important unit, it supports the Quantum Algorithm design and the simulator running.
The Circuit is a class in QuICT, and it uses the qubits and gates to represent the quantum circuit's wires and quantum gates.
In Circuit, it assigns the quantum gates with its wires, and it allows append random gates and the Supremacy circuit. The Circuit
also provides the tools to draw the quantum circuit and translate itself into the OpenQASM file.

How to Build the Circuit in QuICT
---------------------------------
The Circuit provides multiply ways to add gates into the quantum circuit; In QuICT, we can use the operation or(|) to
add gate into the quantum circuit; meanwhile, we can use circuit.append(gate) or circuit.extend(gates) to add gate.


```python
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
```

How to Visualize the Circuit in QuICT
-------------------------------------
There are two ways to visualize the quantum circuit in QuICT, one is to translate the Circuit with the OpenQASM style, the other one is
to draw the graph describing the quantum circuit. By the way, not all quantum gates in QuICT are supported by OpenQASM.


```python 
# qasm
circuit.qasm()

# draw the circuit's graph
circuit.draw(method='matp', filename="QuICT")
```
For further details, please read the examples in example/python or example/demo.


Gates
=====
The quantum gate is designed to operate the qubit's state, it is usually be represented as the 2-dimensional matrix in Quantum Computation.

In QuICT, we use the class BasicGate to achieve the quantum gates, including single/multi-qubits gates and parameters/non-parameters gates.
For each quantum gate in QuICT, it has those properties:
    - name: the name of the gate
    - controls: the control qubits
    - targets: the target qubits
    - params: the parameters
    - matrix: the unitary matrix
    - attributes:
        - is_single: judge whether gate is a one qubit gate
        - is_control_single: judge whether gate has one control bit and one target bit
        - is_diagonal: judge whether gate's matrix is diagonal
        - is_special: judge whether gate's is special gate, such as Measure, Reset, Barrier, Perm, Unitary, ...
    - inverse: inverse itself
    - commutative: decide whether gate is commutative with another gate
    - copy: return a copy of itself
    - qasm: generator OpenQASM string for the gate

Composite Gate
==============
The CompositeGate is the combination of the quantum gates.

The CompositeGate is a class in QuICT, which stores the list of quantum gates. It uses or(|) and xor(^) to append 
the gate or inverse gate, and uses and(&) to remap the control qubits. The main attribute of CompositeGate are
    - append: add gate, can use operation or(|)
    - extend: add list of gates
    - information: the based information about current composite gate.
        - width: the number of qubits
        - size: the number of gates
        - depth: the depth of the composite gate
        - count_1qubit_gate: the number of single-qubit gates
        - count_2qubit_gate: the number of 2-qubits gates
        - count_gate_by_gatetype: the number of the special gates
    - matrix: the integrated matrix of those gates
    - inverse: inverse all gates in composite gate
    - equal: whether is equally with other gate/CompositeGate/Circuit, is judged through the gate matrix.
    - qasm: return the OpenQASM 2.0 describe for the composite gate

Example
-------

```python
from QuICT.core.gate import *

# set gate's attributes
h_gate = H & 1          # create H gate with control qubit 1
cx_gate = CX & [1, 3]   # create CX gate with control qubit 1 and target qubit 3
u2_gate = U2(1, 0)      # create U2 gate with parameters 1 and 0

assert H.is_single()            # H gate is single-qubit gate
assert Measure.is_special()     # Measure gate is a special quantum gate

# create composite gate
cg1 = CompositeGate()

# using default quantum gates
H | cg1(1)                                # append H gate with control qubit 1
cx_gate | cg1                             # append previous cx gate
QFT.build_gate(3) | cg1([0, 3, 4])        # append QFT composite gate with control qubits [0, 3, 4]
U1(1) | cg1(4)                            # append U1 gate with parameters 1 and control qubit 4   

# composite gate information
assert cg1.width() == 5
assert cg1.depth() == 7
assert cg1.size() == 9

# using context to build composite gate
with CompositeGate() as cg_context:
    H & 1
    CX & [1, 3]
    QFT.build_gate(3) & [0, 3, 4]
    U1(1) & 4

assert cg_context.equal(cg1)
```

For further details, please read the examples in example/python or example/demo.


Qubit
=====
The bit is the fundamental concept of classical computation and classical information. The Quantum Computation is built 
upon a similar concept, the quantum bit (qubit).

The qubit also have a state for quantum computation; unlike classical bit, the state can be the combinations of 0 and 1, which
are called superpositions:

:math:`|\mu \rangle \rightarrow \alpha |0 \rangle + \beta |1 \rangle`

In QuICT, we use a data structure Qubit to represent the concept of the qubit in Quantum Computation, and for qubit's state,
use the vector in two-dimensional complex vector space.

The Qubit has two properties:
    - id: the unique ID to distinguish each qubit.
    - measured: store the measurement result of the Qubit after applying the measure gate.

Qureg
=====
The Qureg (Qubit registry) is a data structure in QuICT, which stores a list of Qubits. The Qureg inherits the class List of python,
so that it can be treated as the python list.

The Qureg also allows different initial ways:
    - n(int): build a list of new qubits with given length n.
    - Qubit: build a list with a given qubit.
    - [Qubit/Qureg]: build a list with the given qubits/quregs.

Example
-------

```python
from QuICT.core import Qubit, Qureg

# Qubit
q1 = Qubit()
q2 = Qubit()
assert q1.id != q2.id

# Qureg
qr1 = Qureg(5)          # build qureg with 5 qubits
qr2 = Qureg([q1, q2])   # build qureg with previous qubits
```

For further details, please read the examples in example/python or example/demo.

