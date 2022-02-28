# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# Modification Notice: Code revised for QuICT

# from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.circuit import Circuit
from QuICT.algorithm import SyntheticalUnitary

class TemplateSearching:
# for clifford gates
# search for S, CNOT, H gates

    def __init__(self, qubit_num, gate_num, gate_dep):
        self.qubit_num = qubit_num
        self.gate_num = gate_num
        self.dep = gate_dep
        self.template_list = []
        self.temp_qubit_dep = [0] * qubit_num
        self.target = []
        for i in range(qubit_num):
            self.target.append(i)

    def copy_circuit(self, temp_circuit):
        new_circuit = Circuit(temp_circuit.width())
        new_circuit.extend(temp_circuit.gates)
        return new_circuit

    def identity(self, temp_circuit):
        matrix = SyntheticalUnitary.run(temp_circuit, showSU=False)
        n = np.size(matrix, 0)
        tot = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    tot += abs(matrix[i, j])
                else:
                    tot += abs(matrix[i, j]-1)
        return abs(tot) < 1e-2

    def check(self, temp_circuit):
        n = temp_circuit.size()
        if self.identity(temp_circuit):
            for i in range(n):
                for j in range(1, n - i - 1):
                    new_circuit = self.copy_circuit(temp_circuit.sub_circuit(self.target, i, j))
                    if self.identity(new_circuit):
                        return False
            return True
        return False

    def search(self, temp_gate_num, temp_circuit):

        # check whether it is a template
        if temp_gate_num <= self.gate_num and temp_gate_num >= 1:
            if self.check(temp_circuit):
                self.template_list.append(temp_circuit)
                return
        if temp_gate_num == self.gate_num:
            return

        #brute-force searching
        #s gate
        for s in range(0, self.qubit_num):
            if self.temp_qubit_dep[s] + 1 <= self.dep:
                new_circuit = self.copy_circuit(temp_circuit)
                S | new_circuit(s)
                self.temp_qubit_dep[s] += 1
                self.search(temp_gate_num + 1, new_circuit)
                self.temp_qubit_dep[s] -= 1

        # hadamard gate
        for h in range(0, self.qubit_num):
            if self.temp_qubit_dep[h] + 1 <= self.dep:
                new_circuit = self.copy_circuit(temp_circuit)
                H | new_circuit(h)
                self.temp_qubit_dep[h] += 1
                self.search(temp_gate_num + 1, new_circuit)
                self.temp_qubit_dep[h] -= 1

        # cnot gate
        for cnot_a in range(0, self.qubit_num):
            for cnot_b in range(0, self.qubit_num):
                if cnot_a != cnot_b:
                    temp_max_dep = max(self.temp_qubit_dep[cnot_a], self.temp_qubit_dep[cnot_b]) + 1
                    if temp_max_dep <= self.dep:
                        new_circuit = self.copy_circuit(temp_circuit)
                        CX | new_circuit([cnot_a, cnot_b])
                        temp_dep_a = self.temp_qubit_dep[cnot_a]
                        temp_dep_b = self.temp_qubit_dep[cnot_b]
                        self.temp_qubit_dep[cnot_a] = temp_max_dep
                        self.temp_qubit_dep[cnot_b] = temp_max_dep
                        self.search(temp_gate_num + 1, new_circuit)
                        self.temp_qubit_dep[cnot_a] = temp_dep_a
                        self.temp_qubit_dep[cnot_b] = temp_dep_b

        return

    def run_template_searching(self):
        circuit_temp = Circuit(self.qubit_num)
        self.search(0, circuit_temp)
        return self.template_list

program = TemplateSearching(2, 6, 4)
list_circuit = program.run_template_searching()
print(list_circuit)
j = 48
for item_circuit in list_circuit:
    print(item_circuit.draw('matp', str(j)))
    j += 1