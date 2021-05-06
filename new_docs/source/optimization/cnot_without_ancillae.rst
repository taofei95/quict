CNOT Circuit Optimization without Ancillae
==========================================

You can compact a CNOT circuit without any auxillary qubit using methods
provided with ``QuICT.qcda.optimization``. The input circuit would be
transformed into a ``CompositeGate`` with depth about ``2n ~ 3n``, where
n is the qubit number of input circuit.

Example
-------

.. code:: python

   from QuICT.qcda.optimization import CnotWithoutAncillae

   circuit = Circuit(n)
   circuit.random_append(30 * n, typeList=[GATE_ID["CX"]])
   original_depth = circuit.circuit_depth()
   gates = CnotWithoutAncillae.run(circuit)
   new_depth = gates.circuit_depth()

   print(original_depth)
   print(new_depth)

