#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 8:59 上午
# @Author  : Han Yu
# @File    : topological_cnot_rz.py

from .._optimization import Optimization
from QuICT.QCDA.optimization import topological_cnot
from QuICT.core import *

# topological matrix
TOPO = [[]]

# number of qubits
N = 0

# the input of cnot struct
READ_CNOT = []


def delete_dfs(now):
    """ search for a initial mapping to get better(maybe) topology

    in this order, vertex i is not cut in [i, n)

    Args:
        now(int): the index of vertex now
    """
    global TOPO, N

    delete_dfs.delete_vis[now] = True
    for i in range(N - 1, -1, -1):
        if now != i and TOPO[now][i] and not delete_dfs.delete_vis[i]:
            delete_dfs(i)
    delete_dfs.topo_forward_map[now] = delete_dfs.delete_total
    delete_dfs.topo_backward_map[delete_dfs.delete_total] = now
    delete_dfs.delete_total += 1

def read(circuit):
    """ get describe from the circuit or cnot_struct
    Args:
        circuit(Circuit): the input circuit, contained the information of topology and (maybe) cnot
    Returns:
        list<int>: the inverse of the initial mapping
        list<int>: the Rz_state wait to be deal
        list<float>: the Rz_angle wait to be deal
        set<int>: the index of Rz_state wait to be deal
    """

    global TOPO, READ_CNOT, N
    waitDeal = set()
    N = circuit.circuit_length()
    if len(circuit.topology) == 0:
        TOPO = [[True] * N] * N
    else:
        TOPO = [[False] * N] * N
        for topology in circuit.topology:
            TOPO[topology[0]][topology[1]] = TOPO[topology[1]][topology[0]] = True
    delete_dfs.topo_forward_map = [0] * N
    delete_dfs.topo_backward_map = [0] * N
    delete_dfs.delete_vis = [0] * N
    delete_dfs.delete_total = 0
    delete_dfs(N - 1)

    topo_forward_map = getattr(delete_dfs, "topo_forward_map")
    topo_backward_map = getattr(delete_dfs, "topo_backward_map")

    READ_CNOT = []
    for i in range(N):
        READ_CNOT.append(1 << i)

    termNumber = 0

    cnot_index = dict()
    input = []
    th = []

    for i in range(len(circuit.gates)):
        gate = circuit.gates[i]
        if gate.type() == GATE_ID["CX"]:
            READ_CNOT[topo_forward_map[gate.targ]] ^= \
                    READ_CNOT[topo_forward_map[gate.carg]]
        elif gate.type() == GATE_ID["Rz"]:
            index = cnot_index.setdefault(READ_CNOT[topo_forward_map[gate.targ]], 0)
            if index != 0:
                th[index - 1] += gate.parg
            else:
                termNumber += 1
                cnot_index[READ_CNOT[topo_forward_map[gate.targ]]] = termNumber
                th.append(gate.parg)
                input.append(READ_CNOT[topo_forward_map[gate.targ]])
                waitDeal.add(termNumber - 1)

    return topo_backward_map, input, th, waitDeal

def solve(input, th, waitDeal, undirected_topology):
    """ main part of the algorithm

    Args:
        input(list<int>): the Rz_state wait to be deal
        th(list<float>): the Rz_angle wait to be deal
        waitDeal(set<int>): the index of Rz_state wait to be deal)
        undirected_topology(list<tuple<int, int>>): make topology undirected

    Returns:
        list<CXGates>: the result of the algorithm
    """

    global  N, READ_CNOT

    ans = []
    flag = False
    firstIn = True
    stateChange = []
    for i in range(N):
        stateChange.append(1 << i)

    while len(waitDeal) > 0 or not flag:
        gates = []
        a = [0] * N
        total = 0
        gsxy = []
        needDeal = []
        if len(waitDeal) > 0:
            GateBuilder.setGateType(GATE_ID["Rz"])
            for it in waitDeal:
                val = input[it]
                for i in range(N - 1, -1, -1):
                    if (val & (1 << i)) != 0:
                        if a[i] == 0:
                            a[i] = val
                            break
                        val ^= a[i]

                if val > 0:
                    gsxy.append(input[it])
                    GateBuilder.setTargs(total)
                    GateBuilder.setPargs(th[it])
                    gate = GateBuilder.getGate()
                    gates.append(gate)
                    needDeal.append(it)
                    total += 1
                    if total >= N:
                        break

            for i in range(total):
                waitDeal.remove(needDeal[i])

            for j in range(N):
                val = 1 << j
                for i in range(N - 1, -1, -1):
                    if val & (1 << i) != 0:
                        if a[i] == 0:
                            a[i] = val
                            break
                        val ^= a[i]

                if val > 0:
                    gsxy.append(1 << j)
                    total += 1
                    if total >= N:
                        break
        else:
            for i in range(N):
                gsxy.append(READ_CNOT[i])
                total += 1
            flag = True
        if not firstIn:
            u = []
            v = []
            tempChange = []
            for i in range(N):
                tempChange.append(stateChange[i])
            for i in range(N):
                j = N
                for t in range(i, N):
                    if (1 << i) & tempChange[t]:
                        j = t
                        break
                if j == N:
                    raise Exception("algorithm error")
                if j != i:
                    u.append(j)
                    v.append(i)
                    tempChange[i] ^= tempChange[j]
                for j in range(N):
                    if j != i:
                        if (1 << i) & tempChange[j]:
                            u.append(i)
                            v.append(j)
                            tempChange[j] ^= tempChange[i]
            length = len(u)
            for i in range(length - 1, -1, -1):
                for j in range(N):
                    if gsxy[j] & (1 << v[i]):
                        gsxy[j] ^= 1 << u[i]
        else:
            firstIn = False

        if total == 0:
            break

        if total != N:
            raise Exception("algorithm error")

        print(gsxy, TOPO)
        for gate in gates:
            gate.print_info()

        gsxy_gate = topological_cnot.run_parameter(cnot_struct=gsxy, topology=undirected_topology)
        gsxy_gate.reverse()

        gates.extend(gsxy_gate)

        length = len(gates)
        for j in range(length - 1, -1, -1):
            ans.append(gates[j])
            if gates[j].type() == GATE_ID["CX"]:
                stateChange[gates[j].targ] ^= stateChange[gates[j].carg]

    return ans

class topological_cnot_rz(Optimization):
    """ optimize the cnot_Rz circuit on topological device

    use topological_cnot to optimize a cnot circuit on topological device

    """

    @staticmethod
    def _run(circuit: Circuit, *pargs):
        """
        Args:
            circuit(Circuit): the circuit to be optimize
        """
        global TOPO
        topo_backward_map, input, th, waitDeal = read(circuit)

        if len(circuit.topology) == 0:
            undirected_topology = None
        else:
            undirected_topology = []
            for topology in circuit.topology:
                undirected_topology.append(topology)
                undirected_topology.append((topology[1], topology[0]))

        ans = solve(input, th, waitDeal, undirected_topology)

        if len(circuit.topology) == 0:
            topo = [[True] * N] * N
        else:
            topo = [[False] * N] * N
            for topology in circuit.topology:
                topo[topology[0]][topology[1]] = True

        output = []
        total = 0
        for item in ans:
            if item.type() == GATE_ID["Rz"] or topo[topo_backward_map[item.carg]][topo_backward_map[item.targ]]:
                total += 1
            else:
                total += 5
        for item in ans:
            if item.type() == GATE_ID["CX"]:
                GateBuilder.setGateType(GATE_ID["CX"])
                c = topo_backward_map[item.carg]
                t = topo_backward_map[item.targ]
                if topo[c][t]:
                    GateBuilder.setCargs(c)
                    GateBuilder.setTargs(t)
                    gate = GateBuilder.getGate()
                    output.append(gate)
                else:
                    GateBuilder.setGateType(GATE_ID["H"])
                    GateBuilder.setTargs(c)
                    gate = GateBuilder.getGate()
                    output.append(gate)
                    GateBuilder.setTargs(t)
                    gate = GateBuilder.getGate()
                    output.append(gate)

                    GateBuilder.setGateType(GATE_ID["CX"])
                    GateBuilder.setCargs(t)
                    GateBuilder.setTargs(c)
                    gate = GateBuilder.getGate()
                    output.append(gate)

                    GateBuilder.setGateType(GATE_ID["H"])
                    GateBuilder.setTargs(c)
                    gate = GateBuilder.getGate()
                    output.append(gate)
                    GateBuilder.setTargs(t)
                    gate = GateBuilder.getGate()
                    output.append(gate)
            else:
                GateBuilder.setGateType(GATE_ID["Rz"])
                GateBuilder.setPargs(item.pargs)
                GateBuilder.setTargs(topo_backward_map[item.targ])
                gate = GateBuilder.getGate()
                output.append(gate)
        return output
