Language Translator
==============================

Use OPENQASMInterface to transform between an instance of Circuit and
OpenQASM file.

an example of transform OpenQASM file between an instance of Circuit:

.. code-block:: python
    :linenos:

    from QuICT.tools.interface import OPENQASMInterface

    # load qasm
    qasm = OPENQASMInterface.load_file("../qasm/pea_3_pi_8.qasm")
    if qasm.valid_circuit:
        # generate circuit
        circuit = qasm.circuit
        circuit.print_information()

        new_qasm = OPENQASMInterface.load_circuit(circuit)
        new_qasm.output_qasm("test.qasm")
    else:
        print("Invalid format!")
