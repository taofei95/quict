#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/6/28 下午2:42
# @Author  : Kaiqi Li
# @File    : QFT_Simulator_multigpu

from cupy.cuda import nccl
import numpy as np
import multiprocessing

from QuICT.core import *
from QuICT.algorithm import Amplitude
from QuICT.ops.utils import Proxy
from QuICT.simulation.multigpu_simulator import MultiStateVectorSimulator

from time import time


def build_QFT_circuit(qubit_number, QFT_number):
    circuit = Circuit(qubit_number)

    # for _ in range(3):
    #     QFT.build_gate(qubit_number) | circuit

    for i in range(QFT_number):
        S | circuit(i % qubit_number)

    return circuit


def worker(uid, ndevs, dev_id, qubits, QFT_number):
    proxy = Proxy(ndevs=ndevs, uid=uid, rank=dev_id)

    circuit = build_QFT_circuit(qubits, QFT_number)

    s_time = time()
    simulator = MultiStateVectorSimulator(
        proxy=proxy,
        circuit=circuit,
        precision=np.complex64,
        gpu_device_id=dev_id,
        sync=True
    )
    state = simulator.run()
    e_time = time()

    print(f"finish with {qubits} qubits, spending time {e_time - s_time}")
    print(f"{proxy.rank} with result {state}")


def amplitude_simulator(circuit):
    state_expected = Amplitude.run(circuit)

    print(np.array(state_expected))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")

    uid = nccl.get_unique_id()
    qubits, QFT_n = 5, 5

    p1 = multiprocessing.Process(target=worker, args=(uid, 2, 0, qubits, QFT_n,))
    p2 = multiprocessing.Process(target=worker, args=(uid, 2, 1, qubits, QFT_n,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    circuit = build_QFT_circuit(qubits, QFT_n)
    amplitude_simulator(circuit)
