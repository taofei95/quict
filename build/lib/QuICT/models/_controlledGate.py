from QuICT.models import *
import numpy as np
from itertools import combinations
import itertools
import math
import cmath

TOLERANCE = 1e-12

def _test_parameters(matrix, a, b_half, c_half, d_half):
    U = [[cmath.exp(1j*(a-b_half-d_half))*math.cos(c_half),
          -cmath.exp(1j*(a-b_half+d_half))*math.sin(c_half)],
         [cmath.exp(1j*(a+b_half-d_half))*math.sin(c_half),
          cmath.exp(1j*(a+b_half+d_half))*math.cos(c_half)]]
    return np.allclose(U, matrix, rtol=10*TOLERANCE, atol=TOLERANCE)

def FindParameter(matrix):
    b_half, c_half, d_half = 0, 0, 0
    if abs(matrix[0, 1]) < TOLERANCE:
        two_a = cmath.phase(matrix[0, 0] * matrix[1, 1]) % (2 * math.pi)
        if abs(two_a) < TOLERANCE or abs(two_a) > 2 * math.pi - TOLERANCE:
            a = 0
        else:
            a = two_a / 2.
        d_half = 0  # w.l.g
        b = cmath.phase(matrix[1, 1]) - cmath.phase(matrix[0, 0])
        possible_b_half = [(b / 2.) % (2 * math.pi), (b / 2. + math.pi) % (2 * math.pi)]
        # As we have fixed a, we need to find correct sign for cos(c/2)
        possible_c_half = [0.0, math.pi]
        found = False
        for b_half, c_half in itertools.product(possible_b_half,
                                                possible_c_half):
            if _test_parameters(matrix, a, b_half, c_half, d_half):
                found = True
                break
        if not found:
            raise Exception("Couldn't find parameters for matrix ", matrix,
                            "This shouldn't happen. Maybe the matrix is " +
                            "not unitary?")
    # Case 2: cos(c/2) == 0:
    elif abs(matrix[0, 0]) < TOLERANCE:
        two_a = cmath.phase(-matrix[0, 1] * matrix[1, 0]) % (2 * math.pi)
        if abs(two_a) < TOLERANCE or abs(two_a) > 2 * math.pi - TOLERANCE:
            a = 0
        else:
            a = two_a / 2.
        d_half = 0
        b = cmath.phase(matrix[1, 0]) - cmath.phase(matrix[0, 1]) + math.pi
        possible_b_half = [(b / 2.) % (2 * math.pi), (b / 2. + math.pi) % (2 * math.pi)]
        possible_c_half = [math.pi / 2., 3. / 2. * math.pi]
        found = False
        for b_half, c_half in itertools.product(possible_b_half,
                                                possible_c_half):
            if _test_parameters(matrix, a, b_half, c_half, d_half):
                print(a, b_half, c_half, d_half)
                found = True
                break
        if not found:
            raise Exception("Couldn't find parameters for matrix ", matrix,
                            "This shouldn't happen. Maybe the matrix is " +
                            "not unitary?")
    # Case 3: sin(c/2) != 0 and cos(c/2) !=0:
    else:
        two_a = cmath.phase(matrix[0, 0] * matrix[1, 1]) % (2 * math.pi)
        if abs(two_a) < TOLERANCE or abs(two_a) > 2 * math.pi - TOLERANCE:
            a = 0
        else:
            a = two_a / 2.
        two_d = 2. * cmath.phase(matrix[0, 1]) - 2. * cmath.phase(matrix[0, 0])
        possible_d_half = [two_d / 4. % (2 * math.pi),
                           (two_d / 4. + math.pi / 2.) % (2 * math.pi),
                           (two_d / 4. + math.pi) % (2 * math.pi),
                           (two_d / 4. + 3. / 2. * math.pi) % (2 * math.pi)]
        two_b = 2. * cmath.phase(matrix[1, 0]) - 2. * cmath.phase(matrix[0, 0])
        possible_b_half = [two_b / 4. % (2 * math.pi),
                           (two_b / 4. + math.pi / 2.) % (2 * math.pi),
                           (two_b / 4. + math.pi) % (2 * math.pi),
                           (two_b / 4. + 3. / 2. * math.pi) % (2 * math.pi)]
        tmp = math.acos(abs(matrix[1, 1]))
        possible_c_half = [tmp % (2 * math.pi),
                           (tmp + math.pi) % (2 * math.pi),
                           (-1. * tmp) % (2 * math.pi),
                           (-1. * tmp + math.pi) % (2 * math.pi)]
        found = False
        for b_half, c_half, d_half in itertools.product(possible_b_half,
                                                        possible_c_half,
                                                        possible_d_half):
            if _test_parameters(matrix, a, b_half, c_half, d_half):
                found = True
                break
        if not found:
            raise Exception("Couldn't find parameters for matrix ", matrix,
                            "This shouldn't happen. Maybe the matrix is " +
                            "not unitary?")
    return a, 2 * b_half, 2 * c_half, 2 * d_half

class SingleControlledGate(gateModel):
    def __or__(self, other):
        qureg = self.qureg_trans(other)
        gate_V = np.asmatrix(self.pargs.copy())
        rotation_alpha, rotation_beta, rotation_gamma, rotation_delta = FindParameter(gate_V.reshape(2, 2))

        C = np.asmatrix(Rz(0.5 * (rotation_delta - rotation_beta)).matrix).reshape(2, 2)
        B1 = np.asmatrix(Ry(-0.5 * rotation_gamma).matrix).reshape(2, 2)
        B2 = np.asmatrix(Rz(-0.5 * (rotation_delta + rotation_beta)).matrix).reshape(2, 2)
        B = B1 * B2
        A1 = np.asmatrix(Rz(rotation_beta).matrix).reshape(2, 2)
        A2 = np.asmatrix(Ry(0.5 * rotation_gamma).matrix).reshape(2, 2)
        print(np.round(A1 * B2 * C, decimals=2))
        A = A1 * A2
        # print(np.round(C, decimals=2))
        # print(np.round(B, decimals=2))
        # print(np.round(A, decimals=2))
        print(np.round(A * B * C, decimals=2))

        Rz(0.5 * (rotation_delta - rotation_beta))  | qureg[1]
        CX                                          | (qureg[0], qureg[1])
        Rz(-0.5 * (rotation_delta + rotation_beta)) | qureg[1]
        Ry(-0.5 * rotation_gamma)                   | qureg[1]
        CX                                          | (qureg[0], qureg[1])
        Ry(0.5 * rotation_gamma)                    | qureg[1]
        Rz(rotation_beta)                           | qureg[1]
        U1(rotation_alpha)                          | qureg[0]

    def build_gate(self, other):
        Unitary_gate = self.pargs
        gate_V = np.asmatrix(self.pargs.copy())
        rotation_alpha, rotation_beta, rotation_gamma, rotation_delta = FindParameter(gate_V.reshape(2, 2))
        """
        ABC分解
        """
        gates = []
        GateBuilder.setGateType(GateType.Rz)
        GateBuilder.setPargs(0.5 * (rotation_delta - rotation_beta))
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.Rz)
        GateBuilder.setPargs(-0.5 * (rotation_delta + rotation_beta))
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.Ry)
        GateBuilder.setPargs(-0.5 * rotation_gamma)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.CX)
        GateBuilder.setCargs(other[0])
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.Ry)
        GateBuilder.setPargs(0.5 * rotation_gamma)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.Rz)
        GateBuilder.setPargs(rotation_beta)
        GateBuilder.setTargs(other[1])
        gates.append(GateBuilder.getGate())

        GateBuilder.setGateType(GateType.U1)
        GateBuilder.setPargs(rotation_alpha)
        GateBuilder.setTargs(other[0])
        gates.append(GateBuilder.getGate())

        return gates

SingleCGate = SingleControlledGate()


class MuiltiControlledGate(gateModel):
    def __or__(self, other):
        qureg = self.qureg_trans(other)
        num_qubit = len(qureg)
        gate_V = self.pargs.copy()
        UGate = np.mat(gate_V).reshape(2, 2)
        """
        计算矩阵V = U^{2^{ - (n - 2)}}
        """
        index = 1 / (np.power(2, (num_qubit - 2)))
        e_unitary_vals, e_unitary_vecs = np.linalg.eig(UGate)
        e_vgate_vals = np.power(e_unitary_vals, np.complex(index))
        gate_V = e_unitary_vecs * np.diag(e_vgate_vals) * e_unitary_vecs.I

        gate_VI = gate_V.I.flatten().tolist()[0]
        gate_V = gate_V.flatten().tolist()[0]
        print(gate_V)

        for K in range(1, num_qubit):
            if K == 1:
                items = list(combinations(list(range(1, num_qubit)), K))
                items_int = np.array(items, dtype=int)
                for item in items_int:
                    for j in item:
                        SingleCGate(gate_V) | (qureg[j - 1], qureg[num_qubit - 1])

            if (K > 1) & (K % 2 == 1):
                items = list(combinations(list(range(1, num_qubit)), K))
                items_int = np.array(items, dtype=int)
                for item in items_int:
                    for j in range(1, len(item)):
                        CX | (qureg[item[j - 1] - 1], qureg[item[-1] - 1])
                    SingleCGate(gate_V) | (qureg[item[-1] - 1], qureg[num_qubit - 1])
                    for j in range(1, len(item)):
                        CX | (qureg[item[len(item) - j - 1] - 1], qureg[item[-1] - 1])

            if (K > 1) & (K % 2 == 0):
                items = list(combinations(list(range(1, num_qubit)), K))
                items_int = np.array(items, dtype=int)
                for item in items_int:
                    for j in range(1, len(item)):
                        CX | (qureg[item[j - 1] - 1], qureg[item[-1] - 1])
                    SingleCGate(gate_VI) | (qureg[item[-1] - 1], qureg[num_qubit - 1])
                    for j in range(1, len(item)):
                        CX | (qureg[item[len(item) - j - 1] - 1], qureg[item[-1] - 1])

    def build_gate(self, other):
        gates = []
        num_qubit = len(other)
        gate_V = np.asmatrix(self.pargs.copy()).reshape(2, 2)
        UGate = gate_V
        """
               计算矩阵V = U^{2^{ - (n - 2)}}
        """
        index = 1 / (np.power(2, (num_qubit - 2)))
        e_unitary_vals, e_unitary_vecs = np.linalg.eig(UGate)
        e_vgate_vals = np.power(e_unitary_vals, np.complex(index))
        gate_V = e_unitary_vecs * np.diag(e_vgate_vals) * e_unitary_vecs.I
        gate_VI = gate_V.I.flatten().tolist()[0]
        gate_V = gate_V.flatten().tolist()[0]
        for K in range(1, num_qubit):
            if K == 1:
                items = list(combinations(list(range(1, num_qubit)), K))
                items_int = np.array(items, dtype=int)
                for item in items_int:
                    for j in item:
                        gates.extend(SingleCGate(gate_V).build_gate((other[j - 1], other[num_qubit - 1])))

            if (K > 1) & (K % 2 == 1):
                items = list(combinations(list(range(1, num_qubit)), K))
                items_int = np.array(items, dtype=int)
                for item in items_int:
                    for j in range(1, len(item)):

                        GateBuilder.setGateType(GateType.CX)
                        GateBuilder.setCargs(other[item[j - 1] - 1])
                        GateBuilder.setTargs(other[item[-1] - 1])
                        gates.append(GateBuilder.getGate())

                    SingleCGate(gate_V).build_gate((other[item[-1] - 1], other[num_qubit - 1]))

                    for j in range(1, len(item)):
                        GateBuilder.setGateType(GateType.CX)
                        GateBuilder.setCargs(other[item[len(item) - j - 1] - 1])
                        GateBuilder.setTargs(other[item[-1] - 1])
                        gates.append(GateBuilder.getGate())

            if (K > 1) & (K % 2 == 0):
                items = list(combinations(list(range(1, num_qubit)), K))
                items_int = np.array(items, dtype=int)
                for item in items_int:
                    for j in range(1, len(item)):
                        GateBuilder.setGateType(GateType.CX)
                        GateBuilder.setCargs(other[item[j - 1] - 1])
                        GateBuilder.setTargs(other[item[-1] - 1])
                        gates.append(GateBuilder.getGate())

                    gates.extend(SingleCGate(gate_VI).build_gate((other[item[-1] - 1], other[num_qubit - 1])))

                    for j in range(1, len(item)):
                        GateBuilder.setGateType(GateType.CX)
                        GateBuilder.setCargs(other[item[len(item) - j - 1] - 1])
                        GateBuilder.setTargs(other[item[-1] - 1])
                        gates.append(GateBuilder.getGate())

        return gates

MuiltiCGate = MuiltiControlledGate()

'''
if __name__ == '__main__':
    circuit = Circuit(3)
    gates = MuiltiCGate([0, 1, 1, 0]).build_gate([0, 1])
    circuit.set_flush_gates(gates)
    # CX | circuit
    # MuiltiCGate([1, 0, 0, 1j]) | circuit
    circuit.print_infomation()
    print(np.round(SyntheticalUnitary.run(circuit), decimals=2))
'''