Circuit
=======

.. currentmodule:: QuICT.core.circuit.circuit

.. autoclass:: Circuit
   :show-inheritance:

   .. rubric:: Attributes Summary

   .. autosummary::

      ~Circuit.const_lock
      ~Circuit.fidelity
      ~Circuit.gates
      ~Circuit.id
      ~Circuit.name
      ~Circuit.qubits
      ~Circuit.topology

   .. rubric:: Methods Summary

   .. autosummary::

      ~Circuit.__call__
      ~Circuit.add_topology
      ~Circuit.add_topology_complete
      ~Circuit.append
      ~Circuit.assign_initial_random
      ~Circuit.assign_initial_zeros
      ~Circuit.circuit_count_1qubit
      ~Circuit.circuit_count_2qubit
      ~Circuit.circuit_count_gateType
      ~Circuit.circuit_depth
      ~Circuit.circuit_size
      ~Circuit.circuit_width
      ~Circuit.clear
      ~Circuit.draw
      ~Circuit.exec
      ~Circuit.exec_release
      ~Circuit.extend
      ~Circuit.force_copy
      ~Circuit.index_for_qubit
      ~Circuit.matrix_product_to_circuit
      ~Circuit.partial_prob
      ~Circuit.print_information
      ~Circuit.qasm
      ~Circuit.random_append
      ~Circuit.reset
      ~Circuit.set_exec_gates
      ~Circuit.sub_circuit

   .. rubric:: Attributes Documentation

   .. autoattribute:: const_lock
   .. autoattribute:: fidelity
   .. autoattribute:: gates
   .. autoattribute:: id
   .. autoattribute:: name
   .. autoattribute:: qubits
   .. autoattribute:: topology

   .. rubric:: Methods Documentation

   .. automethod:: __call__
   .. automethod:: add_topology
   .. automethod:: add_topology_complete
   .. automethod:: append
   .. automethod:: assign_initial_random
   .. automethod:: assign_initial_zeros
   .. automethod:: circuit_count_1qubit
   .. automethod:: circuit_count_2qubit
   .. automethod:: circuit_count_gateType
   .. automethod:: circuit_depth
   .. automethod:: circuit_size
   .. automethod:: circuit_width
   .. automethod:: clear
   .. automethod:: draw
   .. automethod:: exec
   .. automethod:: exec_release
   .. automethod:: extend
   .. automethod:: force_copy
   .. automethod:: index_for_qubit
   .. automethod:: matrix_product_to_circuit
   .. automethod:: partial_prob
   .. automethod:: print_information
   .. automethod:: qasm
   .. automethod:: random_append
   .. automethod:: reset
   .. automethod:: set_exec_gates
   .. automethod:: sub_circuit
