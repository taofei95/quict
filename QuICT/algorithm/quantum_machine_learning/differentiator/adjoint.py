import numpy as np

from QuICT.core.gate import *
from QuICT.core.circuit import Circuit
from QuICT.simulation.state_vector import StateVectorSimulator

from QuICT.algorithm.quantum_machine_learning.differentiator import Differentiator


class Adjoint(Differentiator):
    @property
    def grad_vector(self):
        return self._grad_vector

    @grad_vector.setter
    def grad_vector(self, vec):
        self._grad_vector = self._gate_calculator.validate_state_vector(
            vec, self._qubits
        )

    def run(
        self,
        circuit: Circuit,
        state_vector: np.ndarray,
        expectation_op: Union[list, BasicGate],  # Or Hamiltonian?
    ) -> np.ndarray:
        self.initial_circuit(circuit)
        assert state_vector is not None
        self._vector = self._gate_calculator.validate_state_vector(
            state_vector, self._qubits
        )
        # Calculate d(L)/d(|psi_t>)
        self._grad_vector = self._initial_grad_vector(
            state_vector, self._qubits, expectation_op
        )

        for idx in range(len(self._bp_pipeline)):
            origin_gate = self._pipeline[idx]
            gate, qidxes, size = self._bp_pipeline[idx]
            if isinstance(gate, BasicGate):
                # Calculate |psi_t-1>
                self._apply_gate(gate, qidxes, self._vector)

                # Calculate d(L)/d(theta) and write to circuit.gate.pargs.grads
                self._calculate_grad(origin_gate, gate, qidxes)

                # Calculate d(L)/d(|psi_t-1>)
                self._apply_gate(gate, qidxes, self._grad_vector)
            else:
                if size > 1:
                    raise TypeError("Adjoint.run.circuit", "BasicGate".type(gate))
                    # Calculate |psi_t-1>
                    self._apply_compositegate(gate, qidxes, self._vector)

                    # Calculate d(L)/d(theta)

                    # Calculate d(L)/d(|psi_t-1>)
                    self._apply_compositegate(gate, qidxes, self._grad_vector)
                else:
                    raise TypeError("Adjoint.run.circuit", "BasicGate".type(gate))

            # # Calculate |psi_t-1>
            # if size > 1:
            #     self._apply_compositegate(gate, qidxes, self._vector)
            # else:
            #     if isinstance(gate, BasicGate):
            #         self._apply_gate(gate, qidxes, self._vector)
            #     else:
            #         raise TypeError("Adjoint.run.circuit", "BasicGate".type(gate))

            # # Calculate d(L)/d(theta)
            # if gate.variables > 0:
            #     self._calculate_grad(gate, qidxes, size)

            # # Calculate d(L)/d(|psi_t-1>)
            # if size > 1:
            #     self._apply_compositegate(gate, qidxes, self._grad_vector)
            # else:
            #     if isinstance(gate, BasicGate):
            #         self._apply_gate(gate, qidxes, self._grad_vector)
            #     else:
            #         raise TypeError("Adjoint.run.circuit", "BasicGate".type(gate))

    def initial_circuit(self, circuit: Circuit):
        self._qubits = int(circuit.width())
        self._circuit = circuit
        self._bp_circuit = Circuit(circuit.width())
        gates = circuit.gates[::-1]
        for i in range(len(gates)):
            inverse_gate = gates[i].inverse()
            inverse_gate.targs = gates[i].targs
            inverse_gate.cargs = gates[i].cargs
            inverse_gate | self._bp_circuit
        self._pipeline = gates
        self._bp_pipeline = self._bp_circuit.fast_gates
        assert len(self._pipeline) == len(self._bp_pipeline)

    def _validate_expectation_op(self, expectation_op: Union[list, BasicGate]):
        if isinstance(expectation_op, BasicGate):
            expectation_op = [expectation_op]
        if not all(
            [
                (isinstance(op, BasicGate) and op.is_pauli() and len(op.targs) == 1)
                for op in expectation_op
            ]
        ):
            raise Exception  # error
        return expectation_op

    # optimize?
    def _initial_grad_vector(
        self, state_vector, qubits: int, expectation_op: Union[list, BasicGate]
    ):
        state_vector_copy = state_vector.copy()
        expectation_op = self._validate_expectation_op(expectation_op)
        circuit = Circuit(qubits)
        for op in expectation_op:
            op | circuit
        simulator = StateVectorSimulator(
            self._device, self._precision, self._device_id, self._sync
        )
        grad_vector = simulator.run(circuit, state_vector_copy)
        return grad_vector

    def _apply_gate(
        self, gate: BasicGate, qidxes: list, vector, fp: bool = True, parg_id: int = 0
    ):
        gate_type = gate.type
        if gate_type in [GateType.measure, GateType.reset]:
            raise NotImplementedError
        else:
            self._gate_calculator.apply_gate(
                gate, qidxes, vector, self._qubits, fp, parg_id
            )

    def _apply_compositegate(
        self,
        gate: CompositeGate,
        qidxes: list,
        vector,
        fp: bool = True,
        parg_id: int = 0,
    ):
        qidxes_mapping = {}
        cgate_qlist = gate.qubits
        for idx, cq in enumerate(cgate_qlist):
            qidxes_mapping[cq] = qidxes[idx]

        for cgate, cg_idx, size in gate.fast_gates:
            real_qidx = [qidxes_mapping[idx] for idx in cg_idx]
            if size > 1:
                self._apply_compositegate(cgate, real_qidx, vector, fp, parg_id)
            else:
                if isinstance(gate, BasicGate):
                    self._apply_gate(cgate, real_qidx, vector, fp, parg_id)
                else:
                    raise TypeError(
                        "Adjoint.apply_compositegate", "BasicGate".type(gate)
                    )

    def _calculate_grad(self, origin_gate, gate, qidxes: list):
        for i in range(gate.variables):
            vector = self._vector.copy()
            # d(|psi_t>) / d(theta_t^j)
            self._apply_gate(gate, qidxes, vector, fp=False, parg_id=i)
            vector = vector * gate.pargs[i].grads
            # d(L)/d(|psi_t>) * d(|psi_t>) / d(theta_t^j)
            grad = (self._grad_vector @ vector.T).real
            origin_gate.pargs[i].grads = [grad]


if __name__ == "__main__":
    from QuICT.core.gate.utils import Variable
    from QuICT.simulation.utils import GateSimulator
    from QuICT.simulation.state_vector import StateVectorSimulator

    param = Variable(np.array([-3.2]))

    circuit = Circuit(2)
    H | circuit
    Rxx(param[0]) | circuit([0, 1])

    simulator = StateVectorSimulator()
    sv = simulator.run(circuit)

    differ = Adjoint(device="GPU")
    X.targs = [1]
    differ.run(circuit, sv, X)
    # print(differ.vector)
    # print(differ.grad_vector)

