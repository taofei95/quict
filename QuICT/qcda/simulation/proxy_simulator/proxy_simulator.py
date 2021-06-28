import numpy as np
import cupy as cp
import math
import time

from QuICT.core import *
from QuICT.qcda.simulation import BasicSimulator
from QuICT.ops.linalg.proxy import Proxy
from QuICT.qcda.simulation.proxy_simulator.gate_func_single import GateFuncS, GateFuncMS


class ProxySimulator(BasicSimulator):
    def __init__(self, proxy: Proxy, circuit: Circuit, precision, device: int = 0, sync: bool = True):
        self.proxy = proxy
        self._sync = sync
        assert(proxy.rank == device)

        # Get qubits and limitation
        self.total_qubits = int(circuit.circuit_width())
        self.limit_qubits = int(self.total_qubits - np.log2(self.proxy.ndevs))
        self.switch_data = False

        # Initial simulator with limit_qubits
        BasicSimulator.__init__(self, circuit, precision, device, qubits=self.limit_qubits)

        self.initial_vector_state()

        self._algorithm = GateFuncS
        self._proxy_algorithm = GateFuncMS

    def initial_vector_state(self):
        vector_size = 1 << int(self._qubits)
        # Special Case for no gate circuit
        if len(self._gates) == 0:
            self._vector = np.zeros(vector_size, dtype=self._precision)
            if self.proxy.rank == 0:
                self._vector[0] = self._precision(1)
            return

        # Initial qubit's states
        with cp.cuda.Device(self._device):
            self._vector = cp.empty(vector_size, dtype=self._precision)
            if self.proxy.rank == 0:
                self._vector.put(0, self._precision(1))

    def run(self) -> np.ndarray:
        with cp.cuda.Device(self._device):
            for gate in self._gates:
                self.exec(gate)

        return self.vector

    def exec(self, gate):
        matrix = self.get_Matrix(gate)

        if gate.type() == GATE_ID["H"]:
            t_index = self.total_qubits - 1 - gate.targ
            if t_index >= self.limit_qubits:
                self._switch_data(t_index)
                t_index = self.limit_qubits - 1
                self.switch_data = True

            self._algorithm.HGate_matrixdot(
                t_index,
                matrix,
                self._vector,
                self._qubits,
                self._sync
            )
        elif gate.type() == GATE_ID["CRz"]:
            c_index = self.total_qubits - 1 - gate.carg
            t_index = self.total_qubits - 1 - gate.targ

            if c_index >= self.limit_qubits and t_index >= self.limit_qubits:
                if self.proxy.rank & 1 << (c_index - self.limit_qubits):
                    _0_1 = self.proxy.rank & 1 << (t_index - self.limit_qubits)

                    self._proxy_algorithm.CRzGate_matrixdot_pd(
                        _0_1,
                        matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )

            elif c_index >= self.limit_qubits or t_index >= self.limit_qubits:
                if t_index > c_index:
                    _0_1 = self.proxy.rank & 1 << (t_index - self.limit_qubits)
                    self._proxy_algorithm.CRzGate_matrixdot_pc(
                        _0_1,
                        c_index,
                        matrix,
                        self._vector,
                        self._qubits,
                        self._sync
                    )
                else:
                    if self.proxy.rank & 1 << (c_index - self.limit_qubits - 1):
                        
                        self._proxy_algorithm.CRzGate_matrixdot_pt(
                            t_index,
                            matrix,
                            self._vector,
                            self._qubits,
                            self._sync
                        )
            else:
                self._algorithm.CRzGate_matrixdot(
                    c_index,
                    t_index,
                    matrix,
                    self._vector,
                    self._qubits,
                    self._sync
                )

        if self.switch_data:
            self._switch_data(self.total_qubits - 1 - gate.targ)
            self.switch_data = False

    def _switch_data(self, t_index: int):
        """ 
        Switch data with rank(self.rank & t_index >> limitQ) dev; 
        switched by highest index for limit qubit.
        """
        required_qubits = self.limit_qubits - 1
        sending_per_qubits = min(required_qubits, 15)
        sending_size = 1 << sending_per_qubits
        recv_buf = cp.zeros(1 << required_qubits, dtype=self._precision)
        send_buf = self.vector

        destination = self.proxy.rank ^ (1 << (t_index - self.limit_qubits))

        front = self.proxy.rank > destination

        for i in range(math.ceil((1 << required_qubits)/(1 << sending_per_qubits))):
            if front:
                self.proxy.send(send_buf[i*sending_size:(i+1)*sending_size], destination)
            else:
                if i == 0:
                    self.proxy.send(send_buf[-(i+1)*sending_size:], destination)
                else:
                    self.proxy.send(send_buf[-(i+1)*sending_size:-i*sending_size], destination)

            if not front:
                self.proxy.recv(recv_buf[i*sending_size:(i+1)*sending_size], destination)
            else:
                if i == 0:
                    self.proxy.recv(recv_buf[-(i+1)*sending_size:], destination)
                else:
                    self.proxy.recv(recv_buf[-(i+1)*sending_size:-i*sending_size], destination)

        #TODO: Add vector change in simulator
        self.reset_vector(recv_buf, front)

    def reset_vector(self, new_vector, front: bool = True):
        new_vector_size = new_vector.size

        if front:
            self._vector[:new_vector_size] = new_vector
        else:
            self._vector[new_vector_size:] = new_vector
 