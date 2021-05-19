#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/27 2:16 下午
# @Author  : Han Yu
# @File    : statevector_simulator

import numpy as np

from ..disjoint_set import DisjointSet

from .._simulation import BasicSimulator

from QuICT.core import *
from QuICT.ops.linalg.unitary_calculation import *


class StateVectorSimulator(BasicSimulator):
    """ Algorithms to the state vector of a Circuit.

    """

    @classmethod
    def run(cls, circuit: Circuit) -> np.ndarray:
        """
        Get the state vector of circuit

        Args:
            circuit (Circuit): Input circuit to be simulated.

        Returns:
            np.ndarray: The state vector of input circuit.
        """

        qubit = circuit.circuit_width()
        if len(circuit.gates) == 0:
            vector = np.zeros(1 << qubit, dtype=np.complex128)
            vector[0] = 1 + 0j
            return vector
        ordering, small_gates = BasicSimulator.vector_pretreatment(circuit)
        vector = cls.act_unitary_by_ordering(small_gates, ordering)

        return vector

    @classmethod
    def merge_two_unitary(cls, gate_a: BasicGate, gate_b: BasicGate) -> UnitaryGate:
        """
        Combine 2 gates into a new unitary gate.

        Args:
            gate_a (BasicGate): Gate in the left.
            gate_b (BasicGate): Gate in the right.

        Returns:
            UnitaryGate: Combined gate with matrix and affectArgs set properly.
        """

        args_a = gate_a.affectArgs
        args_b = gate_b.affectArgs
        gate_a.print_info()
        gate_b.print_info()

        seta = set(args_a)
        setb = set(args_b)

        if len(seta & setb) == 0:
            args_b.extend(args_a)
            return Unitary(tensor(gate_b.compute_matrix, gate_a.compute_matrix)) & args_b

        setc = seta | setb
        len_a = len(seta)
        len_b = len(setb)
        len_c = len(setc)

        if len_c == len_a:
            mat_a = gate_a.compute_matrix
        else:
            mat_a = MatrixTensorI(gate_a.compute_matrix, 1, 1 << (len_c - len_a))
        if len_c == len_b:
            mat_b = gate_b.compute_matrix
        else:
            mat_b = MatrixTensorI(gate_b.compute_matrix, 1, 1 << (len_c - len_b))

        mps = [0] * len_c
        affectArgs = [0] * len_c
        cnt = len_a
        for rb in range(len_b):
            if args_b[rb] not in seta:
                mps[rb] = cnt
                affectArgs[cnt] = args_b[rb]
                cnt += 1
            else:
                for ra in range(len_a):
                    if args_a[ra] == args_b[rb]:
                        mps[rb] = ra
                        affectArgs[ra] = args_b[rb]
                        break
        cnt = len_b
        for ra in range(len_a):
            if args_a[ra] not in setb:
                mps[cnt] = ra
                affectArgs[ra] = args_a[ra]
                cnt += 1
        mat_b = MatrixPermutation(mat_b, np.array(mps))
        print(args_b)
        print(args_a)
        print(affectArgs)
        gate = Unitary(dot(mat_b, mat_a)) & affectArgs
        print(gate.affectArgs)
        return gate

    @classmethod
    def act_unitary_by_ordering(cls, gates: List[BasicGate], ordering: List[int], qubit) -> np.ndarray:
        """
        act a gate sequence into a 2^n state vector. The combination order is determined by
        input parameter.

        Args:
            gates (List[BasicGate]): A list consisting of n gates to be merged.
            ordering (List[int]): A permutation of [0,n-1] denoting the combination order of gates.
                with some negative numbers inserted. -(stick+1) means act the leftest unitary at stick-th
                position on the state vector.
                If number i is at the j-th position, i-th merge operation would combine 2 gates
                around j-th seam (Remember that those 2 gates might have already been merged into larger
                gates).
            qubit (int): the number of qubits

        Returns:
            np.ndarray: state vector
        """

        vector = np.zeros(1 << qubit, dtype=np.complex128)
        vector[0] = 1 + 0j
        len_gate = len(gates)

        dSet = DisjointSet(len_gate)
        if len(ordering) < len(gates):
            assert 0
        for order in ordering:
            if order < 0:
                stick = -order - 1
                order_stick = dSet.find(stick)
                gateA = gates[order_stick]
                affectArgs = gateA.affectArgs
                if gateA.targets + gateA.controls < qubit:
                    mat_a = MatrixTensorI(gateA.compute_matrix, 1, 1 << (qubit - (gateA.targets + gateA.controls)))
                else:
                    mat_a = gateA.compute_matrix
                for j in range(qubit):
                    if j not in affectArgs:
                        affectArgs.append(j)
                if affectArgs != [i for i in range(qubit)]:
                    mat_a = MatrixPermutation(mat_a, np.array(affectArgs))
                vector = dot(mat_a, vector)
            else:
                order_left = dSet.find(order)
                order_right = dSet.find(order + 1)
                gateA = gates[order_left]
                gateB = gates[order_right]
                x = dSet.union(order_left, order_right)
                gates[x] = StateVectorSimulator.merge_two_unitary(gateA, gateB)
        return vector


from QuICT.ops.linalg.gpu_calculator_cupy import GPUCalculatorCP

calc = GPUCalculatorCP()


class StateVectorSimulatorRefine:

    @staticmethod
    def run(circuit: Circuit, initial_state: np.ndarray) -> np.ndarray:
        state = initial_state.copy()

        for gate in circuit.gates:
            gate: BasicGate
            cur_state = calc.vectordot(gate.compute_matrix, state, False)
            del state
            state = cur_state
        return state.get()
