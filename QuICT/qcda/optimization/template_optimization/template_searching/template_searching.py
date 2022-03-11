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

from QuICT.core.gate import *
from QuICT.core.circuit import Circuit
from QuICT.algorithm import SyntheticalUnitary
import time

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

    def check_minimum(self, temp_circuit):
        n = temp_circuit.size()
        if self.identity(temp_circuit):
            for i in range(n):
                for j in range(1, n - i):
                    new_circuit = self.copy_circuit(temp_circuit.sub_circuit(self.target, i, j))
                    if self.identity(new_circuit):
                        return False
            return True
        return False

    def graph_isomorphism(self, graph_a, graph_b, single_list_a, single_list_b):

        # check if an ordered-list graph is not isomorphism with the other
        # first only check cnot circuit
        flag_all_mapping = True
        for vertex_a in range(self.qubit_num):
            if self.mapping[vertex_a] == -1:
                flag_all_mapping = False
                break

        if flag_all_mapping:
            flag_check_graph = True
            for i in range(self.qubit_num):
                j = self.mapping[i]
                if len(graph_a[i]) == len(graph_b[j]) and len(single_list_a[i]) == len(single_list_b[j]):
                    for k in range(len(graph_a[i])):
                        if not flag_check_graph:
                            break
                        check_a = graph_a[i][k]
                        check_b = graph_b[j][k]
                        flag_check_graph = flag_check_graph & (self.mapping[check_a] == check_b)
                    for k in range(len(single_list_a[i])):
                        if not flag_check_graph:
                            break
                        check_a = single_list_a[i][k]
                        check_b = single_list_b[j][k]
                        if len(check_a) == len(check_b):
                            for p in range(len(check_a)):
                                flag_check_graph = flag_check_graph & (self.mapping[check_a[p].targ] == check_b[p].targ)
                                flag_check_graph = flag_check_graph & (check_a[p].type == check_b[p].type)
                        else:
                            flag_check_graph = False
                else:
                    flag_check_graph = False
                    break
            return flag_check_graph
        else:
            flag_check_graph = False
            for i in range(self.qubit_num):
                if not self.mapped[i]:
                    if len(graph_a[vertex_a]) == len(graph_b[i]):
                        flag_check_list = True
                        self.mapped[i] = True
                        self.mapping[vertex_a] = i
                        temp_mapped = []
                        temp_mapping = []
                        for j in range(len(graph_a[vertex_a])):
                            check_a = graph_a[vertex_a][j]
                            check_b = graph_b[i][j]
                            temp_mapping.append(self.mapping[check_a])
                            temp_mapped.append(self.mapped[check_b])
                            if self.mapping[check_a] != -1 and self.mapping[check_a] != check_b:
                                flag_check_list = False
                            else:
                                self.mapping[check_a] = check_b
                                self.mapped[check_b] = True

                        if flag_check_list:
                            flag_check_graph = flag_check_graph or self.graph_isomorphism(graph_a, graph_b, single_list_a, single_list_b)
                            if flag_check_graph:
                                break

                        if not flag_check_graph:
                            self.mapped[i] = False
                            self.mapping[vertex_a] = -1
                            for j in range(len(graph_a[vertex_a])):
                                check_a = graph_a[vertex_a][j]
                                check_b = graph_b[i][j]
                                self.mapping[check_a] = temp_mapping[j]
                                self.mapped[check_b] = temp_mapped[j]

            return flag_check_graph

    def check_circuit_not_isomorphism(self, circuit_a, circuit_b):

        # check if a circuit is not isomorphism with the other
        # first only check cnot circuit
        cnot_graph_a = []
        single_list_a = []
        cnot_graph_b = []
        single_list_b = []
        for i in range(self.qubit_num):
            cnot_graph_a.append([])
            single_list_a.append([])
            cnot_graph_b.append([])
            single_list_b.append([])

        temp_list = []
        for i in range(self.qubit_num):
            temp_list.append([])
        for gate_a in circuit_a.gates:
            if gate_a.is_control_single():
                cnot_graph_a[gate_a.carg].append(gate_a.targ)
                single_list_a[gate_a.carg].append(temp_list[gate_a.carg])
                temp_list[gate_a.carg] = []
            else:
                temp_list[gate_a.targ].append(gate_a)
        for i in range(self.qubit_num):
            if temp_list[i] != []:
                single_list_a[i].append(temp_list[i])

        temp_list = []
        for i in range(self.qubit_num):
            temp_list.append([])
        for gate_b in circuit_b.gates:
            if gate_b.is_control_single():
                cnot_graph_b[gate_b.carg].append(gate_b.targ)
                single_list_b[gate_b.carg].append(temp_list[gate_b.carg])
                temp_list[gate_b.carg] = []
            else:
                temp_list[gate_b.targ].append(gate_b)
        for i in range(self.qubit_num):
            if temp_list[i] != []:
                single_list_b[i].append(temp_list[i])

        self.mapping = []
        self.mapped = []
        for i in range(self.qubit_num):
            self.mapping.append(-1)
            self.mapped.append(False)
        flag_carg_graph = self.graph_isomorphism(cnot_graph_a, cnot_graph_b, single_list_a, single_list_b)

        cnot_graph_a = []
        single_list_a = []
        cnot_graph_b = []
        single_list_b = []
        for i in range(self.qubit_num):
            cnot_graph_a.append([])
            single_list_a.append([])
            cnot_graph_b.append([])
            single_list_b.append([])

        temp_list = []
        for i in range(self.qubit_num):
            temp_list.append([])
        for gate_a in circuit_a.gates:
            if gate_a.is_control_single():
                cnot_graph_a[gate_a.targ].append(gate_a.carg)
                single_list_a[gate_a.targ].append(temp_list[gate_a.targ])
                temp_list[gate_a.targ] = []
            else:
                temp_list[gate_a.targ].append(gate_a)
        for i in range(self.qubit_num):
            if temp_list[i] != []:
                single_list_a[i].append(temp_list[i])

        temp_list = []
        for i in range(self.qubit_num):
            temp_list.append([])
        for gate_b in circuit_b.gates:
            if gate_b.is_control_single():
                cnot_graph_b[gate_b.targ].append(gate_b.carg)
                single_list_b[gate_b.targ].append(temp_list[gate_b.targ])
                temp_list[gate_b.targ] = []
            else:
                temp_list[gate_b.targ].append(gate_b)
        for i in range(self.qubit_num):
            if temp_list[i] != []:
                single_list_b[i].append(temp_list[i])

        self.mapping = []
        self.mapped = []
        for i in range(self.qubit_num):
            self.mapping.append(-1)
            self.mapped.append(False)
        flag_targ_graph = self.graph_isomorphism(cnot_graph_a, cnot_graph_b, single_list_a, single_list_b)

        return not (flag_targ_graph and flag_carg_graph)

    def check_list_not_isomorphism(self, temp_circuit):

        # check if the circuit is not isomorphism with every template in the list
        no_isomorphism = True
        for template in self.template_list:
            no_isomorphism = no_isomorphism & self.check_circuit_not_isomorphism(temp_circuit, template)
        return no_isomorphism

    def search(self, temp_gate_num, temp_circuit):

        # check whether it is a template
        if 1 <= temp_gate_num <= self.gate_num:
            if self.identity(temp_circuit):
                # if self.check_list_not_isomorphism(temp_circuit):
                #     self.template_list.append(temp_circuit)
                self.template_list.append([temp_circuit, self.check_minimum(temp_circuit)])
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
        len_list = len(self.template_list)
        relationship_iso = [-1] * len_list
        for i in range(len(self.template_list)):
            if relationship_iso[i] == -1:
                relationship_iso[i] = i
                for j in range(len(self.template_list)):
                    if relationship_iso[j] == -1 and not self.check_circuit_not_isomorphism(self.template_list[i][0], self.template_list[j][0]):
                            self.template_list[i][1] = self.template_list[i][1] & self.template_list[j][1]
                            relationship_iso[j] = i
                            self.template_list[j][1] = False
        return self.template_list


print(time.perf_counter())
program = TemplateSearching(3, 4, 4)
list_circuit = program.run_template_searching()
print(time.perf_counter())
label = 1
for item_circuit in list_circuit:
    if item_circuit[1]:
        print(item_circuit[0].draw('matp', str(label)))
        label += 1
