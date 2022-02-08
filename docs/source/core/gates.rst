Gates
=====
The quantum gate is designed to operate the qubit's state, it usually be represent as the
2-dimensional matrix in the Quantum Computation.

In QuICT, we use the class BasicGate to achieve the quantum gates, including single/multi qubit gates and parameters/non-parameters gates.
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

The CompositeGate is a class in QuICT, which store the list of quantum gates. It uses or(|) and xor(^) to append 
the gate or inverse gate, and uses and(&) to remapping the control qubits. The main attributes of CompositeGate is
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
=======

.. code-block:: python

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
    assert cg1.depth() == 5
    assert cg1.size() == 5

    # using context to build composite gate
    with CompositeGate() as cg_context:
        H & 1
        CX & [1, 3]
        QFT.build_gate(3) & [0, 3, 4]
        U1(1) & 4

    assert cg_context.equal(cg1)

For further details, please read the examples in example/python or example/demo.
