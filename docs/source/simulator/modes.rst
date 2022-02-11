State-Vector Simulator
======================
The state-vector simulator holds the qubits' states during running the quantum circuit. After running
through the given quantum circuit, it returns the final qubits' states.

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
The unitary simulator is used to generate the unitary matrix of the given quantum circuit. Not like the other
classical simulator, the unitary simulator do not care about the qubits' states, it only returns the unitary matrix
of the quantum algorithm.

Example
>>>>>>>

.. code:: ipython3

    # Initial unitary simulator
    unitary_simulator = Simulator(
        device="CPU",
        backend="unitary",
        shots=1
    )
    result = unitary_simulator.run(circuit)    # get simulation's result

.. parsed-literal::

    {'id': '440329748b0b11ec80d963c17826bae3',
     'device': 'CPU',
     'backend': 'unitary',
     'shots': 1,
     'options': {'precision': 'double'},
     'spending_time': 3.682887315750122,
     'output_path': '/home/likaiqi/Workplace/test/QuICT/example/demo/output/440329748b0b11ec80d963c17826bae3',
     'counts': defaultdict(int, {}),
     'unitary_matrix': array([[-6.07232039e-17+2.87850229e-17j,  8.88063244e-17-1.55236501e-17j,
              1.76776695e-01+1.76776695e-01j, ...,
              7.13834983e-17+9.42059954e-17j, -1.76776695e-01+1.76776695e-01j,
             -1.76776695e-01+1.76776695e-01j],
            [ 7.43843391e-17-4.25606797e-17j,  9.71523969e-17-3.44101078e-17j,
             -1.76776695e-01-1.76776695e-01j, ...,
             -7.56947743e-17-5.53745984e-17j, -1.76776695e-01+1.76776695e-01j,
              1.76776695e-01-1.76776695e-01j],
            [ 1.76776695e-01-1.76776695e-01j,  1.76776695e-01-1.76776695e-01j,
              6.67745909e-17+7.27432107e-18j, ...,
             -1.76776695e-01-1.76776695e-01j,  9.50988899e-17-3.51585673e-17j,
             -4.68753295e-17+1.20031377e-17j],
            ...,
            [ 4.40522111e-17-3.92478329e-18j, -4.15901801e-17+1.27601331e-18j,
             -1.76776695e-01-1.76776695e-01j, ...,
              4.07748908e-17-2.12508660e-17j, -1.76776695e-01+1.76776695e-01j,
             -1.76776695e-01+1.76776695e-01j],
            [ 1.76776695e-01-1.76776695e-01j, -1.76776695e-01+1.76776695e-01j,
             -2.14497404e-17-1.36656436e-17j, ...,
              1.76776695e-01+1.76776695e-01j,  5.54468159e-17+1.99827349e-17j,
              7.01963462e-17-1.29716533e-18j],
            [ 1.76776695e-01-1.76776695e-01j,  1.76776695e-01-1.76776695e-01j,
             -2.25892009e-17-9.82634769e-18j, ...,
              1.76776695e-01+1.76776695e-01j, -6.80407082e-17-7.37645205e-18j,
              1.80014454e-17+2.29079844e-17j]])}


Multi-GPU State-Vector Simulator
================================
During the incresment of the qubits, the required memory of simulation is exponential increasing. The Multi-GPU State-Vector simulator
is designed to use multi-gpus in one machine to simulate the running of the quantum circuit; therefore, the simulator can be faster and more extensive.

Example
>>>>>>>

.. code:: python

    from concurrent.futures import ProcessPoolExecutor, as_completed
    from cupy.cuda import nccl

    from QuICT.utility import Proxy
    from QuICT.simulation.gpu_simulator import MultiStateVectorSimulator

    def worker_thread(ndev, uid, dev_id):
        # Using multi-GPU simulator
        proxy = Proxy(ndevs=ndev, uid=uid, dev_id=dev_id)
        simulator = MultiStateVectorSimulator(
            proxy=proxy,
            precision="double",
            gpu_device_id=dev_id,
            sync=True
        )
        state = simulator.run(cir)

        return state

    if __name__ == "__main__":
        ndev = 2    # Device number
        uid = nccl.get_unique_id()    # generate nccl id for Proxy connection
        with ProcessPoolExecutor(max_workers=ndev) as executor:
            tasks = [
                executor.submit(worker_thread, ndev, uid, dev_id) for dev_id in range(ndev)
            ]

        # Collect result from each device
        results = []
        for t in as_completed(tasks):
            results.append(t.result())

Remote Simulator
================
Currently the QuICT supports to simulate with the simulator from other platform (Qiskit and QCompute).

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
