#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/12/2 下午9:58
# @Author  : Han Yu
# @File    : sabre

import copy
from typing import List, Optional

from QuICT.core.layout import *
from QuICT.core.gate import *
from QuICT.core.circuit import *
from QuICT.qcda.utility import OutputAligner

class DAGNode:
    def __init__(self, gate):
        self.gate = gate
        self.bits = gate.cargs + gate.targs
        self.edges = []
        self.attach = []
        self.pre_number = 0

class SABREMapping:
    def __init__(
        self,
        layout : Layout,
        sizeE = 20,
        w = 0.5,
    ):
        self._layout = layout
        self._sizeE  = sizeE
        self._w      = w
        self.phy2logic = None

    @OutputAligner()
    def execute(
            self,
            circuit: Union[Circuit, CompositeGate],
            l2p: List[int] = None
    ) -> Union[Circuit, CompositeGate]:
        SIZE_E = self._sizeE
        W = self._w
        layout = self._layout
        decay_parameter = 0.001
        decay_cycle = 5

        nodes = []
        qubit_number = circuit.width()
        phy_number = layout.qubit_number
        D: List[List[int]] = []
        for _ in range(phy_number):
            D.append([phy_number for _ in range(phy_number)])

        for edge in layout.edge_list:
            D[edge.u][edge.v] = D[edge.v][edge.u] = 1
        for i in range(phy_number):
            for j in range(phy_number):
                if j == i:
                    continue
                for k in range(phy_number):
                    if k == i or k == j:
                        continue
                    D[i][j] = min(D[i][j], D[i][k] + D[k][j])

        predag: List[Optional[DAGNode]] = [None for _ in range(qubit_number)]
        F = []
        if l2p is None:
            l2p = [i for i in range(phy_number)]
        else:
            goal = set([i for i in range(phy_number)])
            for number in l2p:
                goal.remove(number)
            while len(l2p) < phy_number:
                l2p.append(goal.pop())
        p2l = [0 for _ in range(phy_number)]
        for index in l2p:
            p2l[l2p[index]] = index

        def can_execute(node):
            # print(node.bits)
            if len(node.bits) == 1:
                return True
            elif len(node.bits) == 2:
                # print(layout.check_edge(l2p[node.bits[0]], l2p[node.bits[1]]))
                return layout.check_edge(l2p[node.bits[0]], l2p[node.bits[1]])
            else:
                assert False

        def obtain_swaps():
            candidates = []
            bits = set()
            for node in F:
                if len(node.bits) == 1:
                    continue
                bits = bits.union(set([l2p[bit] for bit in node.bits]))
            for edge in layout.edge_list:
                if edge.u in bits or edge.v in bits:
                    candidates.append(edge)
            return candidates

        def temp_pi(edge):
            new_mapping = copy.deepcopy(l2p)
            u, v = edge.u, edge.v
            # print(p2l)
            # print(u, v)
            new_mapping[p2l[u]] = v
            new_mapping[p2l[v]] = u
            return new_mapping

        def phy_gate(g):
            _g = g.copy()
            _g.cargs = [l2p[carg] for carg in g.cargs]
            _g.targs = [l2p[targ] for targ in g.targs]
            return _g

        def H(newl2p):
            H_basic = 0
            H_extend = 0
            F_count = len(F)
            E_queue = []
            for node in F:
                H_basic += D[newl2p[node.bits[0]]][newl2p[node.bits[1]]] / F_count
                E_queue.append(node)

            ESet = []
            decQueue = []
            while len(ESet) < SIZE_E and len(E_queue) > 0:
                node = E_queue.pop(0)
                decQueue.append(node)
                for succ in node.edges:
                    succ.pre_number -= 1
                    assert succ.pre_number >= 0
                    if succ.pre_number == 0:
                        ESet.append(succ)
                        E_queue.append(succ)

            E_count = len(ESet)
            for node in ESet:
                H_extend += D[newl2p[node.bits[0]]][newl2p[node.bits[1]]] / E_count

            for node in decQueue:
                for n in node.edges:
                    n.pre_number += 1

            return H_basic + W * H_extend

        exe_gates = []

        for gate in circuit.gates:
            node = DAGNode(gate)
            nodes.append(node)
            pre_number = 0
            if len(node.bits) == 1:
                dag = predag[node.bits[0]]
                # print(gate, dag)
                # print(gate, dag)
                if dag is not None:
                    dag.attach.append(node)
                else:
                    exe_gates.append(phy_gate(node.gate))
            else:
                for bit in node.bits:
                    dag = predag[bit]
                    if dag is not None:
                        if len(node.bits) == 0:
                            dag.attach.append(node)
                        elif node not in dag.edges:
                            dag.edges.append(node)
                            pre_number += 1
                for bit in node.bits:
                    predag[bit] = node

            node.pre_number = pre_number
            if len(node.bits) == 2 and pre_number == 0:
                F.append(node)

        decay = [1 for _ in range(phy_number)]
        decay_time = 0
        while len(F) > 0:
            decay_time += 1
            if decay_time % decay_cycle == 0:
                decay = [1 for _ in range(phy_number)]
            exe_gate_list = []
            for node in F:
                if can_execute(node):
                    exe_gate_list.append(node)
                    exe_gates.append(phy_gate(node.gate))
                    nodes.remove(node)
                    for gate in node.attach:
                        nodes.remove(gate)
                        exe_gates.append(phy_gate(gate.gate))
            if len(exe_gate_list) != 0:
                for node in exe_gate_list:
                    F.remove(node)
                    for succ in node.edges:
                        succ.pre_number -= 1
                        if succ.pre_number < 0:
                            assert False
                        if succ.pre_number == 0:
                            F.append(succ)
                continue
            else:
                candidate_list = obtain_swaps()
                theSwap = None
                theScore = 0
                thePi = None
                for swap in candidate_list:
                    pi = temp_pi(swap)
                    H_score = H(pi)
                    # print(swap.u, swap.v, p2l[swap.u], p2l[swap.v])
                    H_score = H_score * max(decay[p2l[swap.u]], decay[p2l[swap.v]])
                    if theSwap is None or H_score < theScore:
                        theScore = H_score
                        theSwap = swap
                        thePi = pi
                p2l[theSwap.u], p2l[theSwap.v] = p2l[theSwap.v], p2l[theSwap.u]
                l2p = thePi
                # print("time")
                exe_gates.append((Swap & ([theSwap.u, theSwap.v])))
                decay[p2l[theSwap.u]] += decay_parameter
                decay[p2l[theSwap.v]] += decay_parameter
        circuit = Circuit(len(l2p))
        for g in exe_gates:
            g | circuit
        self.phy2logic = p2l
        return circuit
