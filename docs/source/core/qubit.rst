Qubit
=====
The bit is the fundamental concept of classical computation and classical information. The Quantum Computation is build 
upon a similar concept, the quantum bit (qubit).

The qubit also have a state for quantum computation; unlike classical bit, the state can be the combinations of 0 and 1, which
often called superpositions:

:math:`|\mu \rangle \rightarrow \alpha |0 \rangle + \beta |1 \rangle`

In QuICT, we use a data structure Qubit to represent the concept of qubit in Quantum Computation, and for qubit's state, use the
vector in two-dimensional complex vector space.

The Qubit has two properties:
    - id: the unique ID to distinguish each qubit.
    - measured: store the measure result of the Qubit after apply the measure gate.

Qureg
=====
The Qureg (Qubit registry) is a data structure in QuICT, which store a list of Qubits.

The Qureg inherits the class List of python, so that it can be treated as python list. 
The Qureg also allows different initial ways:
    - n(int): build a list of new qubits with given length n.
    - Qubit: build a list with given qubit.
    - [Qubit/Qureg]: build a list with the given qubits/quregs.

Example
-------

.. code-block:: python

    from QuICT.core import Qubit, Qureg
    
    
    # Qubit
    q1 = Qubit()
    q2 = Qubit()
    assert q1.id != q2.id

    # Qureg
    qr1 = Qureg(5)          # build qureg with 5 qubits
    qr2 = Qureg([q1, q2])   # build qureg with previous qubits

For further details, please read the examples in example/python or example/demo.
