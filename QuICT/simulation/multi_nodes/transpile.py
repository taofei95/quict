import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.utils import GateType
from QuICT.simulation.utils import GateGroup, GATE_TYPE_to_ID
from QuICT.qcda.synthesis import GateDecomposition


# HALF_SWAP_GATE = [
#     [H, SX, SY, SW, U2, U3, Rx, Ry],
#     [CH, CU3],
#     [Fsim],
#     [Rxx, Ryy]
# ]
# ALL_SWAP_GATE = [
#     X,
#     Y,
#     [CX, CY],
#     swap
# ]
# CTARGS_SWAP_GATE = [
#     [CX, CY],
#     [CH, CU3],
#     [Fsim],
#     [Rxx, Ryy],
#     swap,
#     unitary
# ]
# Special_gate = [
#     measure,
#     reset
# ]
ALL_PENTY = 0.5
HALF_PENTY = 1
CARG_PENTY = 2
PROB_ADD = 0.5


NORMAL_GATE_SET_1qubit = [
    GateType.h, GateType.sx, GateType.sy, GateType.sw,
    GateType.u2, GateType.u3, GateType.rx, GateType.ry
]



class Transpile:
    def __init__(self, ndev: int):
        self._ndev = ndev
        self._split_qb_num = int(np.log2(ndev))
        assert 2 ** self._split_qb_num == self._ndev

    def _transpile(self, circuit: Circuit, split_qubits: list):
        t_qubits = circuit.qubits - self._split_qb_num
        transpiled_circuit = [Circuit(t_qubits, i) for i in range(self._ndev)]
        common_cgate: CompositeGate = CompositeGate()
        for gate in circuit.gates:
            gate_args = gate.cargs + gate.targs
            if len(set(gate_args) & set(split_qubits)) == 0:
                # gate_args replace
                gate & self._args_adjust(gate_args, split_qubits)
                gate | common_cgate
                continue

            # gate influence in split qubits
            # append common composite gate into each transpiled_circuit
            if common_cgate.size():
                for tcir in transpiled_circuit:
                    common_cgate | tcir

                common_cgate.clean()

            for index, tcir in enumerate(transpiled_circuit):
                if len(gate_args) == 1:
                    self._single_qubit_gate_transpile(index, gate, split_qubits)
                elif len(gate_args) == 2:
                    self._double_qubits_gate_transpile(index, gate, split_qubits)
                else:
                    raise ValueError("Only supportted transpilt for gate less than 2 qubits.")

    def _single_qubit_gate_transpile(self, index: int, gate: BasicGate, split_qubits: list):
        arg = gate.targ
        split_idx = split_qubits.index(arg)
        _0_1 = index & (1 << split_idx)

        gate_type = gate.type
        splited_cgate = CompositeGate()
        if gate_type in NORMAL_GATE_SET_1qubit:
            destination = index ^ (1 << split_idx)
            # splited_cgate add op.half_switch(destination)
        elif gate_type:
            pass

    def _double_qubits_gate_transpile(self, index: int, gate: BasicGate, split_qubits: list):
        pass

    def _args_adjust(self, gate_args, split_args):
        res = gate_args[:]
        for i in range(len(gate_args)):
            garg = gate_args[i]
            for sarg in split_args:
                if sarg < garg:
                    res[i] -= 1

        return res

    def _split_qubits(self, circuit: Circuit) -> list:
        qubits = circuit.width()
        comm_cost = np.array([0] * qubits, dtype=np.float32)
        for gate in circuit.gates:
            # Consider trigger here
            gate_type = gate.type
            gate_args = gate.cargs + gate.targs
            if gate_type in GATE_TYPE_to_ID[GateGroup.matrix_1arg]:
                comm_cost[gate_args[0]] += HALF_PENTY
            elif gate_type in [GateType.x, GateType.y]:
                comm_cost[gate_args[0]] += ALL_PENTY
            elif gate_type in [
                GateType.cx, GateType.cy, GateType.ch, GateType.cu3,
                GateType.fsim, GateType.Rxx, GateType.Ryy, GateType.swap,
                GateType.unitary
            ]:
                for arg in gate_args:
                    comm_cost[arg] += CARG_PENTY
            elif gate_type in [GateType.measure, GateType.reset]:
                comm_cost[arg] += PROB_ADD
            else:
                continue

        split_qubits = []
        for _ in range(self._split_qb_num):
            min_idx = np.argmin(comm_cost)
            split_qubits.append(min_idx)
            comm_cost[min_idx] = np.Infinity

        return split_qubits
    
    def run(self, circuit: Circuit):
        # step 1: run gatedecomposition
        circuit.gates = GateDecomposition.execute(circuit).gates

        # step 2: decided split qubits
        split_qubits = self._split_qubits(circuit)

        # step 3: transpile circuit by split qubits [add op.data_swap, change gate]
        transpiled_circuits = self._transpile(circuit, split_qubits)

        return transpiled_circuits
