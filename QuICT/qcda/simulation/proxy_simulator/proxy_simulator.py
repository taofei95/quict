import numpy as np
import cupy as cp
import copy
import time

from QuICT.core import *
from QuICT.ops.linalg.proxy import Proxy


class ProxySimulator:
    def __init__(self, proxy: Proxy, simulator):
        self.proxy = proxy
        self.simulator = simulator

        assert(proxy.rank == simulator.device)

        # Get qubits and limitation
        self.qubits = copy.deepcopy(simulator.qubits)
        self.limit_qubits = int(self.qubits - np.log2(self.proxy.ndevs))
        self.switch_data = False

        # Reset simulator qubit size, and initial simulator
        self.simulator.qubits = self.limit_qubits

        self.initial_vector_state()

    def initial_vector_state(self):
        self.simulator.initial_vector_state()

        if self.proxy.rank != 0:
            self.simulator.vector[0] = np.complex64(0)

    def run(self) -> np.ndarray:
        with cp.cuda.Device(self.proxy.rank):
            for gate in self.simulator.gates:
                self._calculate(gate)

        return self.simulator.vector

    def _calculate(self, gate):
        if gate.type() == GATE_ID["H"]:
            t_index = self.qubits - 1 - gate.targ
            if t_index >= self.limit_qubits:
                self._switch_data(t_index)
                gate.targs = 0
                self.switch_data = True

            self.simulator.exec(gate)

        elif gate.type() == GATE_ID["CRz"]:
            cindex = gate.carg
            tindex = gate.targ

            if cindex >= self.limit_qubits and tindex >= self.limit_qubits:
                if self.proxy.rank & 1 << (cindex - self.limit_qubits):
                    params = {"_0_1": self.proxy.rank & 1 << (tindex - self.limit_qubits)}

                    self.simulator.calculator(gate, params)

            elif cindex >= self.limit_qubits or tindex >= self.limit_qubits:
                if cindex > tindex:
                    _0_1 = self.proxy.rank & 1 << (cindex - self.limit_qubits)

                    self.simulator.calculator(gate, _0_1, cindex)
                else:
                    if self.proxy.rank & 1 << (cindex - self.limit_qubits):
                        params = {
                            "tindex": tindex,
                        }

                        self.simulator.calculator(gate, None, params)
            else:
                self.simulator.exec(gate)

        if self.switch_data:
            self._switch_data(t_index)
            self.switch_data = False

    def _switch_data(self, t_index: int):
        """ 
        Switch data with rank(self.rank & t_index >> limitQ) dev; 
        switched by highest index for limit qubit.
        """
        start_time = time.time()

        buf_size = 1 << (self.limit_qubits - 1)
        recv_buf = cp.zeros(buf_size, dtype=self.simulator.precision)
        send_buf = self.simulator.vector

        destination = self.proxy.rank ^ (1 << (t_index - self.limit_qubits))

        front = self.proxy.rank > destination

        if front:
            self.proxy.send(send_buf[:buf_size], destination)
        else:
            self.proxy.send(send_buf[buf_size:], destination)

        self.proxy.recv(recv_buf, destination)

        #TODO: Add vector change in simulator
        self.simulator.reset_vector(recv_buf, front)

        print(f"data switch time: {time.time() - start_time}")
 