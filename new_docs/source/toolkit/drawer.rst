Circuit Drawer
=====================

Use Drawer to draw quantum circuits, an example is:

.. code-block:: python
    :linenos:

    circuit = Circuit(5)
    circuit.random_append(50)
    circuit.draw(method='matp')
    circuit.draw(method='command')
    circuit.draw(method='tex')

the methods name include:

- matp: generate an JPEG image

- command: generate an text image

- tex: generate tex code
