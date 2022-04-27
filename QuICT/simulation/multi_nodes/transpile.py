from QuICT.core import Circuit
from QuICT.core.utils import GateType


HALF_SWAP_GATE = [
    [H, SX, SY, SW, U2, U3, Rx, Ry],
    [CH, CU3],
    [Fsim],
    [Rxx, Ryy]
]
ALL_SWAP_GATE = [
    X,
    Y,
    [CX, CY],
    swap
]
CTARGS_SWAP_GATE = [
    [CX, CY],
    [CH, CU3],
    [Fsim],
    [Rxx, Ryy],
    swap,
    unitary
]
Special_gate = [
    measure,
    reset
]


class Transpile:
    def __init__(self, ndev: int):
        self._ndev = ndev

    def _transpile(self):
        pass
    
    def _split_qubits(self, circuit: Circuit) -> list:
        qubits = circuit.width()
        comm_cost = [0] * qubits
        for gate in circuit.gates:
            # Consider trigger here
            continue
        
        return None
    
    def run(self, circuit: Circuit):
        # step 1: run gatedecomposition
        # step 2: decided split qubits
        # step 3: transpile circuit by split qubits [add op.data_swap, change gate]
        
        pass
