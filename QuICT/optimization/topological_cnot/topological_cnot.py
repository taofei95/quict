#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/27 1:42 下午
# @Author  : Han Yu
# @File    : topological_cnot.py

import numpy as np
from queue import Queue

from .._optimization import Optimization
from QuICT.models import *

# topological matrix
from ...models._gate import GATE_ID

TOPO = [[]]

# number of qubits
N = 0

# the input of cnot struct
READ_CNOT = []

# the the gates which makes the identity of "read_cnot", the ans is its inverse
GATES = []

class Steiner_Tree(object):
    """ the Steiner_Tree struct

    we use exponent time to construct the exact steiner tree
    # fix it with a 2-approximation polynomial algorithm

    Attributes:
        N(int): the number of vertexes in the graph.
        matrix(n * n boolean matrix): topological relationship for n vertexes,
                                      it is undirected.
        dp(n * 2^n np.array): dp array to struct the tree.
                dp[i][j] means the root is i, state is j's subtree.
                when subtree contain vertex k, j & (1 << k) != 0
        ST(list<int>): the indexes of vertexes in tree.
        father(list<int>): the father of vertexes in the tree.
        sons(list<int>): the son of vertexes in the tree.
        pre(np.array<int>): the state which update this state in dp.
        root(int): the root of the tree.
    """

    def __init__(self, n, topo):
        """
        Args:
            n(int): number of the vertexes in the tree
            topo(n * n boolean matrix): topological relationship for n vertexes,
                                      it is directed.
        """
        self.N = n
        self.matrix = [[False] * n] * n
        self.dp = np.array([], dtype=np.int)
        self.ST = []
        for i in range(n):
            for j in range(i + 1, n):
                self.matrix[i][j] = self.matrix[j][i] = topo[i][j] or topo[j][i]
        self.father = []
        self.sons = []
        self.pre = np.array([], dtype=np.int)
        self.root = 0

    def build_ST(self, ST_input : list, lower_bound):
        """ build Steiner_Tree with ST_input in it

        ST_input[-1] must be the root of the tree.

        Args:
            ST_input(list<int>): the indexes of vertexes which should be contained in the tree.
                                 ST_input[-1] should be the root.
            lower_bound(int): the minimum index of vertexes which can be used

        """

        size = len(ST_input)
        self.root = ST_input[-1]
        self.dp = np.array([-1] * self.N * (1 << size), dtype=np.int).reshape((self.N, 1 << size))
        self.pre = np.zeros((self.N, 1 << size, 2), dtype=np.int)
        self.ST = [0] * self.N
        self.father = [-1] * self.N
        self.sons = []
        for i in range(self.N):
            self.sons.append([])
        now = 0
        for i in range(size):
            self.ST[ST_input[i]] = 1 << i
            now |= self.ST[ST_input[i]]
        for i in range(lower_bound, self.N):
            self.dp[i][self.ST[i]] = 0

        que = Queue()
        for j in range(1 << size):
            vis = [0] * self.N
            for i in range(lower_bound, self.N):
                if self.ST[i] != 0 and (self.ST[i] & j) == 0:
                    continue
                sub = j & (j - 1)
                while sub != 0:
                    subx = self.ST[i] | sub
                    suby = self.ST[i] | (j - sub)
                    if self.dp[i][subx] != -1 and self.dp[i][suby] != -1:
                        temp = self.dp[i][subx] + self.dp[i][suby]
                        if self.dp[i][sub] == -1 or temp < self.dp[i][sub]:
                            self.dp[i][sub] = temp
                            self.pre[i][sub][0] = subx
                            self.pre[i][sub][1] = suby

                    sub = (sub - 1) & j
                if self.dp[i][j] != -1:
                    que.put(i)
                    vis[i] = True

            while not que.empty():
                u = que.get()
                vis[u] = False
                for i in range(lower_bound, self.N):
                    if self.matrix[i][u] and self.dp[u][j] != -1:
                        temp = self.dp[u][j] + 1
                        if self.dp[i][self.ST[i] | j] == -1 or self.dp[i][self.ST[i] | j] > temp:
                            self.dp[i][self.ST[i] | j] = temp
                            self.pre[i][self.ST[i] | j][:] = [u - self.N, j]
                            if (self.ST[i] | j) != j or vis[i]:
                                continue
                            vis[i] = True
                            que.put(i)
        self.build_STtree(self.root, now)

    def build_STtree(self, root, state):
        """ build the tree with the dp and pre array

        Args:
            root(int): the i in dp[i][j]
            state(int): the j in dp[i][j]
        """

        if state == self.ST[root]:
            return
        _pre = self.pre[root][state]
        if _pre[0] == 0:
            return
        if _pre[0] > 0:
            self.build_STtree(root, _pre[0])
            self.build_STtree(root, _pre[1])
        else:
            self.sons[root].append(_pre[0] + self.N)
            self.father[_pre[0] + self.N] = root
            self.build_STtree(_pre[0] + self.N, _pre[1])

    def elimination_below(self, gauss_elimination : list):
        """ elimination with some rows below ith row and ith column.

        Args:
            gauss_elimination(list<int>): the equations should be changed in the process.
        """

        self._elimination_below_dfs(self.root, gauss_elimination)

    def _elimination_below_dfs(self, now, gauss_elimination : list):
        """ elimination with dfs

        Args:
            now(int): vertex now
            gauss_elimination(list<int>): the equations should be changed in the process.

        """
        for son in self.sons[now]:
            if self.ST[son] == 0:
                GateBuilder.setCargs(now)
                GateBuilder.setTargs(son)
                gauss_elimination[son] ^= gauss_elimination[now]
                gate = GateBuilder.getGate()
                GATES.append(gate)
            self._elimination_below_dfs(son, gauss_elimination)
        if now != self.root:
            GateBuilder.setCargs(self.father[now])
            GateBuilder.setTargs(now)
            gate = GateBuilder.getGate()
            gauss_elimination[now] ^= gauss_elimination[self.father[now]]
            GATES.append(gate)

    def elimination_above(self, gauss_elimination : list):
        self._elimination_above_preorder(self.root, gauss_elimination)
        self._elimination_above_postorder(self.root, gauss_elimination)

    def _elimination_above_preorder(self, now, gauss_elimination : list):
        for son in self.sons[now]:
            if self.ST[son] == 0:
                GateBuilder.setCargs(son)
                GateBuilder.setTargs(now)
                gate = GateBuilder.getGate()
                gauss_elimination[now] ^= gauss_elimination[self.father[now]]
                GATES.append(gate)

        for son in self.sons[now]:
            self._elimination_above_preorder(son, gauss_elimination)

    def _elimination_above_postorder(self, now, gauss_elimination: list):
        for son in self.sons[now]:
            self._elimination_above_postorder(son, gauss_elimination)
        if now != self.root:
            GateBuilder.setTargs(self.father[now])
            GateBuilder.setCargs(now)
            gauss_elimination[self.father[now]] ^= gauss_elimination[now]
            gate = GateBuilder.getGate()
            GATES.append(gate)

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

def read(circuit, cnot_struct, topology):
    """ get describe from the circuit or cnot_struct
    Args:
        circuit(Circuit): the input circuit, contained the information of topology and (maybe) cnot
        cnot_struct(list<int>): the information of cnot. if None, the information is contained in the circuit
        topology(list<tuple<int, int>>): topology of circuit, None or empty means fully connected
    Returns:
        Steiner_Tree: the whole graph of the st
        list<int>: the inverse of the initial mapping
    """

    global TOPO, READ_CNOT, N
    if circuit is not None:
        N = circuit.circuit_length()
        if len(circuit.topology) == 0:
            TOPO = [[True] * N] * N
        else:
            TOPO = [[False] * N] * N
            for topology in circuit.topology:
                TOPO[topology[0]][topology[1]] = TOPO[topology[1]][topology[0]] = True
    else:
        N = len(cnot_struct)
        if topology is None or len(topology) == 0:
            TOPO = [[True] * N] * N
        else:
            TOPO = [[False] * N] * N
            for topos in topology:
                TOPO[topos[0]][topos[1]] = TOPO[topos[1]][topos[0]] = True

    delete_dfs.topo_forward_map = [0] * N
    delete_dfs.topo_backward_map = [0] * N
    delete_dfs.delete_vis = [0] * N
    delete_dfs.delete_total = 0

    delete_dfs(N - 1)

    topo_forward_map = getattr(delete_dfs, "topo_forward_map")
    topo_backward_map = getattr(delete_dfs, "topo_backward_map")

    if cnot_struct is not None:
        READ_CNOT = cnot_struct
    else:
        READ_CNOT = []
        for i in range(N):
            READ_CNOT.append(1 << i)

        for i in range(len(circuit.gates)):
            gate = circuit.gates[i]
            if gate.type() == GATE_ID["CX"]:
                READ_CNOT[topo_forward_map[gate.targ]] ^= \
                        READ_CNOT[topo_forward_map[gate.carg]]

    ST = Steiner_Tree(N, TOPO)
    return ST, topo_backward_map

def solve(ST_tree):
    """ main part of the algorithm

    Args:
        ST_tree(Steiner_Tree): the whole graph

    Returns:
        list<CXGates>: the result of the algorithm

    """
    global GATES, N, READ_CNOT

    # apply Gaussian Elimination on the matrix
    gauss_elimination = READ_CNOT
    GATES = []

    for i in range(N):
        j = N
        for t in range(i, N):
            if gauss_elimination[t] & (1 << i):
                j = t
                break

        if j >= N:
            raise Exception("the matrix is not singular matrix")

        # find a line that the ith bit is 1, and find a path to make the ith bit of ith line 1
        pre = [-1] * N
        if i != j:
            bfs = Queue()
            bfs.put(j)
            while not bfs.empty():
                u = bfs.get()
                for j in range(i, N):
                    if j != u and TOPO[u][j]:
                        if pre[j] == -1:
                            pre[j] = u
                            if j == i:
                                break
                            bfs.put(j)
                if pre[i] != -1:
                    break

        paths = [i]
        u = pre[i]
        while u != -1:
            paths.append(u)
            u = pre[u]

        for j in range(len(paths) - 2, -1, -1):
            GateBuilder.setCargs(paths[j + 1])
            GateBuilder.setTargs(paths[j])
            gauss_elimination[paths[j]] ^= gauss_elimination[paths[j + 1]]
            gate = GateBuilder.getGate()
            GATES.append(gate)

        # elimination below rows
        needCover = []
        for j in range(i + 1, N):
            if gauss_elimination[j] & (1 << i):
                needCover.append(j)
        if len(needCover) > 0:
            needCover.append(i)
            ST_tree.build_ST(needCover, i)
            ST_tree.elimination_below(gauss_elimination)

        # elimination this row

        # find a set S whose summation in equal to row 1 except column i
        back_gauss_elimination = [0] * (i + 1)
        xor_result = [0] * (i + 1)
        for j in range(i + 1, N):
            back_gauss_elimination.append(gauss_elimination[j])
            xor_result.append(1 << j)

        for j in range(i + 1, N):
            k = N
            for t in range(j, N):
                if back_gauss_elimination[t] & (1 << j):
                    k = t
                    break
            if k == N:
                raise Exception("the matrix is not singular matrix")
            back_gauss_elimination[k], back_gauss_elimination[j] = \
                back_gauss_elimination[j], back_gauss_elimination[k]
            xor_result[k], xor_result[j] = xor_result[j], xor_result[k]
            for k in range(j + 1, N):
                if back_gauss_elimination[k] & (1 << j):
                    back_gauss_elimination[k] ^= back_gauss_elimination[j]
                    xor_result[k] ^= xor_result[j]

        val = gauss_elimination[i]
        S_element = 0
        for j in range(i + 1, N):
            if val & (1 << j):
                val ^= back_gauss_elimination[j]
                S_element ^= xor_result[j]

        needCover = []
        for j in range(i + 1, N):
            if S_element & (1 << j):
                needCover.append(j)

        if len(needCover) > 0:
            needCover.append(i)
            ST_tree.build_ST(needCover, i)
            ST_tree.elimination_above(gauss_elimination)

    GATES.reverse()
    return GATES

class topological_cnot(Optimization):
    """ optimize the cnot circuit on topological device
    https://arxiv.org/pdf/1910.14478.pdf

    use steiner tree to optimize a cnot circuit on topological device

    """

    @classmethod
    def run_parameter(cls, cnot_struct, topology):
        """ optimize the circuit

        Args:
            cnot_struct(list<int>): the struct of cnot circuit
            topology(list<tuple<int, int>>): topology of circuit

        """
        return cls._run(cnot_struct = cnot_struct, topology = topology)

    @staticmethod
    def _run(circuit : Circuit = None, cnot_struct = None, topology = None):
        """
        Args:
            circuit(Circuit): the circuit to be optimize
            cnot_struct(list<int>/None): the struct of cnot circuit. if None, read circuit
            topology(list<tuple<int, int>>): topology of circuit
        """
        global TOPO, N
        GateBuilder.setGateType(GATE_ID["CX"])
        steiner_tree, topo_backward_map = read(circuit, cnot_struct, topology)
        ans = solve(steiner_tree)

        if circuit is not None:
            N = circuit.circuit_length()
            if len(circuit.topology) == 0:
                topo = [[True] * N] * N
            else:
                topo = [[False] * N] * N
                for topology in circuit.topology:
                    topo[topology[0]][topology[1]] = topo[topology[1]][topology[0]] = True
        else:
            N = len(cnot_struct)
            if topology is None or len(topology) == 0:
                topo = [[True] * N] * N
            else:
                topo = [[False] * N] * N
                for topos in topology:
                    topo[topos[0]][topos[1]] = topo[topos[1]][topos[0]] = True

        output = []
        for item in ans:
            c = topo_backward_map[item.carg]
            t = topo_backward_map[item.targ]
            if topo[c][t]:
                GateBuilder.setGateType(GATE_ID["CX"])
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
        return output
