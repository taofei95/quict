Synthetical Unitary
===================

The Synthetical Unitary algorithm is designed to calculate the unitary matrix 
of the quantum circuit.

example
-------

.. code-block:: python

    import numpy as np

    from QuICT.core import Circuit
    from QuICT.algorithm import SyntheticalUnitary


    # Build circuit with 100 random gates and 5 qubits
    circuit = Circuit(5)
    circuit.random_append(rand_size=100)

    unitary = SyntheticalUnitary.run(circuit, showSU=False)
