The Design Philosophy of QuICT
-----------------------------------------------------------------------
QuICT platform is a tool to describe, compile and execute quantum computer.
In QuICT's Design Philosophy, the compiling and execution of the quantum circuit
is separated.

- Compiling: Design the quantum circuit with some special gate set. The input of the compiling can be some unitary or quantum circuit. QuICT devote itself to design an executable circuit with as small cost as possible in physical layout. Then, QuICT output IR(Intermediate representation) to the part of execution.

- Execution: Execute the quantum circuit with classical simulator or actual quantum computers.

Key Components
-----------------------------------------------------------------------

.. figure:: ./images/code_en.png
   :width: 1000px

According to the Design Philosophy of QuICT, QuICT's code and Components is
divide in several parts:

- Algorithm Library: it provides classical quantum algorithm(Shor, Grover and so on) and quantum simulation algorithm/application

- Quantum Circuit Design Automation: it provide the data struction of quantum circuit, compiling process(including synthesis, optimization and mapping) and classical simulation.

- Toolkit: it provide toolkit for users, including drawer of circuit, translator of other languages and so on.

Quick Start
-------------

.. code-block:: python

    from QuICT.core import Circuit, H, X, Measure, PermFx

    def deutsch_jozsa_main_oracle(f, qreg, ancilla):
        PermFx(f) | (qreg, ancilla)

    def run_deutsch_jozsa(f, n, oracle):
        """ an oracle, use Deutsch_Jozsa to decide whether f is balanced

        f(list): the function to be decided
        n(int): the input bit
        oracle(function): oracle function
        """

        # Determine number of qreg
        circuit = Circuit(n + 1)

        # start the eng and allocate qubits
        qreg = circuit([i for i in range(n)])
        ancilla = circuit(n)

        # Start with qreg in equal superposition and ancilla in |->
        H | qreg
        X | ancilla
        H | ancilla

        # Apply oracle U_f which flips the phase of every state |x> with f(x) = 1
        oracle(f, qreg, ancilla)

        # Apply H
        H | qreg
        # Measure
        Measure | qreg
        Measure | ancilla

        circuit.exec()

        y = int(qreg)

        if y == 0:
            print('Function is constant. y={}'.format(y))
        else:
            print('Function is balanced. y={}'.format(y))

    if __name__ == '__main__':
        test_number = 5
        test = [0, 1] * 2 ** (test_number - 1)
        run_deutsch_jozsa(test, test_number, deutsch_jozsa_main_oracle)

Contents
----------

.. toctree::
    :maxdepth: 2
    :caption: Installaion

    install/package.rst
    install/docker.rst

.. toctree::
    :maxdepth: 2
    :caption: Algorithm

    algorithm/grover.rst
    algorithm/shor.rst
    algorithm/synthetical_unitary.rst
    algorithm/weight_decision.rst

.. toctree::
    :maxdepth: 2
    :caption: Core and Workflow

    workflow/gates.rst
    workflow/circuit.rst
    workflow/qubit.rst
    workflow/QCDA.rst

.. toctree::
   :maxdepth: 2
   :caption: Synthesis

   synthesis/gate_decomposition.rst
   synthesis/gate_transform.rst
   synthesis/unitary_transform.rst
   synthesis/arithmetic_circuit.rst

.. toctree::
   :maxdepth: 2
   :caption: Optimization

   optimization/commutative_optimization.rst
   optimization/cnot_without_ancillae.rst
   optimization/template_optimization.rst

.. toctree::
   :maxdepth: 2
   :caption: Mapping

   mapping/1D-mapping.rst

.. toctree::
   :maxdepth: 2
   :caption: simulator

   simulator/amplitude.rst
   simulator/unitary.rst
   simulator/statevector.rst
   simulator/proxy.rst

.. toctree::
   :maxdepth: 2
   :caption: Toolkit

   toolkit/drawer.rst
   toolkit/translator.rst
   toolkit/CLI.rst
   toolkit/UI.rst

.. toctree::
   :maxdepth: 2
   :caption: API-Documents

   api/QuICT.rst
