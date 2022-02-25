State-Vector Simulator
======================
The state-vector simulator holds the qubits' states during running the quantum circuit. After running
through the given quantum circuit, it returns the measurement of the quantum circuit.

Example
>>>>>>>

.. code:: ipython3

    from QuICT.core import Circuit
    from QuICT.core.utils import GateType
    from QuICT.simulation import Simulator

    # Build circuit with 100 random gates and 5 qubits
    circuit = Circuit(5)
    type_list = [GateType.x, GateType.h, GateType.rx, GateType.ry, GateType.rz, GateType.cx]
    circuit.random_append(rand_size=100, typelist=type_list)
    
    # Initial Simulator
    simulator = Simulator(
        device="GPU",
        backend="statevector",
        shots=10,
        precision="double"
    )
    result = simulator.run(circuit)    # get simulation's result

.. parsed-literal::

    {'id': '1778fbd88b0911ecb845233b8af251ab',
     'device': 'GPU',
     'backend': 'statevector',
     'shots': 10,
     'options': {'precision': 'double',
      'gpu_device_id': 0,
      'sync': False,
      'optimize': False},
     'spending_time': 0.2988075494766236,
     'output_path': '~/QuICT/example/demo/output/1778fbd88b0911ecb845233b8af251ab',
     'counts': defaultdict(int,
                 {'10010': 2,
                  '00000': 1,
                  '11111': 1,
                  '11001': 1,
                  '00110': 1,
                  '01101': 1,
                  '11010': 1,
                  '11100': 1,
                  '10001': 1})}



Unitary Simulator
======================
The unitary simulator is split into two steps to simulate the quantum circuit. First, it calculates
the unitary matrix of the given quantum circuit. After that, the unitary simulator uses the linear
operation to calculate the qubits' state vector dot with the unitary matrix and uses measure operation
to generate the final qubits' state.

Example
>>>>>>>

.. code:: ipython3

    # Initial unitary simulator
    unitary_simulator = Simulator(
        device="CPU",
        backend="unitary",
        shots=10
    )
    result = unitary_simulator.run(circuit)    # get simulation's result

.. parsed-literal::

    {'id': '9cc0e0b0960f11eca0429bf5d59c4d03',
     'device': 'CPU',
     'backend': 'unitary',
     'shots': 10,
     'options': {'precision': 'double'},
     'spending_time': 0.39297690391540524,
     'counts': defaultdict(int,
                 {'10110': 1,
                 '10100': 1,
                 '01111': 1,
                 '00111': 1,
                 '01001': 2,
                 '10111': 1,
                 '10000': 1,
                 '01100': 1,
                 '01000': 1}),
     'output_path': '/home/likaiqi/Workplace/test/QuICT/example/demo/output/9cc0e0b0960f11eca0429bf5d59c4d03'}


Multi-GPU State-Vector Simulator
================================
During the increment of the qubits, the required memory of simulation is exponentially increasing. The Multi-GPU State-Vector simulator
is designed to use multi-GPUs in one machine to simulate the running of the quantum circuit; therefore, the simulator can be faster and more extensive.

Example
>>>>>>>

.. code:: python

    # Initial multi-GPU simulator
    multi_simulator = Simulator(
        device="GPU",
        backend="multiGPU",
        shots=10,
        ndev=2
    )
    result = multi_simulator.run(circuit)    # get simulation's result


Remote Simulator
================
Currently, the QuICT supports to simulate with the simulator from other platforms (Qiskit and QCompute).

Example
>>>>>>>

.. code:: ipython3

    # Initial remote simulator
    simulator = Simulator(
        device="qcompute",
        backend="cloud_baidu_sim2_earth",
        shots=10,
        token=qcompute_token
    )
    result = simulator.run(circuit)    # get simulation's result
