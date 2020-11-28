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
TOPO = [[]]

# number of qubits
N = 0

# the input of cnot struct
READ_CNOT = []

# the the gates which makes the indentity of "read_cnot", the ans is its inverse
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

    def solve0(self, gauss_elimination : list):
        GateBuilder.setGateType(GateType.CX)
        self.solve0_dfs(self.root, gauss_elimination)

    def solve0_dfs(self, now, gauss_elimination : list):
        for son in self.sons[now]:
            if self.ST[son] == 0:
                GateBuilder.setCargs(now)
                GateBuilder.setTargs(son)
                gauss_elimination[son] ^= gauss_elimination[now]
                gate = GateBuilder.getGate()
                GATES.append(gate)
            self.solve0_dfs(son, gauss_elimination)
        if now != self.root:
            GateBuilder.setCargs(self.father[now])
            GateBuilder.setTargs(now)
            gate = GateBuilder.getGate()
            gauss_elimination[now] ^= gauss_elimination[self.father[now]]
            GATES.append(gate)

    def solve1(self, gauss_elimination : list):
        GateBuilder.setGateType(GateType.CX)
        self.solve1_dfs0(self.root, gauss_elimination)
        self.solve1_dfs2(self.root, gauss_elimination)

    def solve1_dfs0(self, now, gauss_elimination : list):
        for son in self.sons[now]:
            self.solve1_dfs0(son, gauss_elimination)

        if self.ST[now] != 0:
            for son in self.sons[now]:
                if self.ST[son] == 0:
                    GateBuilder.setTargs(now)
                    GateBuilder.setCargs(son)
                    gauss_elimination[now] ^= gauss_elimination[son]
                    gate = GateBuilder.getGate()
                    GATES.append(gate)
                    self.solve1_dfs1(son, gauss_elimination)

    def solve1_dfs1(self, now, gauss_elimination: list):
        for son in self.sons[now]:
            if self.ST[son] == 0:
                GateBuilder.setTargs(now)
                GateBuilder.setCargs(son)
                gauss_elimination[now] ^= gauss_elimination[son]
                gate = GateBuilder.getGate()
                GATES.append(gate)
                self.solve1_dfs1(son, gauss_elimination)

    def solve1_dfs2(self, now, gauss_elimination: list):
        for son in self.sons[now]:
            self.solve1_dfs2(son, gauss_elimination)
        if now != self.root:
            GateBuilder.setTargs(self.father[now])
            GateBuilder.setCargs(now)
            gauss_elimination[self.father[now]] ^= gauss_elimination[now]
            gate = GateBuilder.getGate()
            GATES.append(gate)

ST = None

def delete_dfs(now):
    """ search for a initial mapping to get better(maybe) topology

    Args:
        now(int): the index of vertex now
    """
    global TOPO, N

    if not hasattr(delete_dfs, 'topo_forward_map'):
        delete_dfs.topo_forward_map = [0] * N
    if not hasattr(delete_dfs, 'topo_backward_map'):
        delete_dfs.topo_backward_map = [0] * N
    if not hasattr(delete_dfs, 'delete_vis'):
        delete_dfs.delete_vis = [0] * N
    if not hasattr(delete_dfs, 'delete_total'):
        delete_dfs.delete_total = 0

    delete_dfs.delete_vis[now] = True
    for i in range(N - 1, -1, -1):
        if now != i and TOPO[now][i] and not delete_dfs.delete_vis[i]:
            delete_dfs(i)
    delete_dfs.topo_forward_map[now] = delete_dfs.delete_total
    delete_dfs.topo_backward_map[delete_dfs.delete_total] = now
    delete_dfs.delete_total += 1

def read(circuit, cnot_struct):
    """ get describe from the circuit or cnot_struct
    Args:
        circuit(Circuit): the input circuit, contained the information of topology and (maybe) cnot
        cnot_struct(list<int>): the information of cnot. if None, the information is contained in the circuit

    Returns:
        list<int>: the inverse of the initial mapping
    """

    global TOPO, READ_CNOT, N, ST
    N = circuit.circuit_length()
    if len(circuit.topology) == 0:
        TOPO = [[True] * N] * N
    else:
        TOPO = [[False] * N] * N
        for topology in circuit.topology:
            TOPO[topology[0]][topology[1]] = TOPO[topology[1]][topology[0]] = True
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
            if gate.type() == GateType.CX:
                READ_CNOT[topo_forward_map[gate.targ]] ^= \
                        READ_CNOT[topo_forward_map[gate.carg]]
        ST = Steiner_Tree(N, TOPO)

    return topo_backward_map

def solve():
    """ main part of the algorithm

    Returns:
        list<CXGates>: the result of the algorithm

    """
    global GATES, N, READ_CNOT

    ans = []

    # apply Gaussian Elimination on the matrix
    gauss_elimination = READ_CNOT

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
                for j in range(N):
                    if j != u and TOPO[u][j]:
                        if pre[j] == -1:
                            pre[j] = u
                            if j == i:
                                break
                            bfs.put(j)
                if pre[i] != -1:
                    break

        Step_begin = len(GATES)
        u = pre[i]
        target = i
        GateBuilder.setGateType(GateType.CX)
        while u != -1:
            GateBuilder.setCargs(u)
            GateBuilder.setTargs(target)
            gate = GateBuilder.getGate()
            GATES.append(gate)
            gauss_elimination[target] ^= gauss_elimination[u]
            target = u
            u = pre[u]
        Step_end = len(GATES)

        needCover = []
        cover = 0
        for j in range(i + 1, N):
            if gauss_elimination[j] & (1 << i):
                needCover.append(j)
                cover += 1
        if cover > 0:
            needCover.append(i)
            cover += 1
            ST.build_ST(needCover, i)
            ST.solve0(gauss_elimination)

        # 寻找回消
        fz_gsxy = [0] * (i + 1)
        xor_result = [0] * (i + 1)
        for j in range(i + 1, N):
            fz_gsxy.append(gauss_elimination[j])
            xor_result.append(1 << j)

        for j in range(i + 1, N):
            k = N
            for t in range(j, N):
                if fz_gsxy[t] & (1 << j):
                    k = t
                    break
            if k == N:
                raise Exception("ERROR2")
            fz_gsxy[k], fz_gsxy[j] = fz_gsxy[j], fz_gsxy[k]
            xor_result[k], xor_result[j] = xor_result[j], xor_result[k]
            for k in range(j + 1, N):
                if fz_gsxy[k] & (1 << j):
                    fz_gsxy[k] ^= fz_gsxy[j]
                    xor_result[k]  ^= xor_result[j]

        val = gauss_elimination[i]
        result_xy = 0
        for j in range(i + 1, N):
            if val & (1 << j):
                val ^= fz_gsxy[j]
                result_xy ^= xor_result[j]

        needCover = []
        cover = 0
        for j in range(i + 1, N):
            if result_xy & (1 << j):
                needCover.append(j)
                cover += 1

        if cover > 0:
            needCover.append(i)
            cover += 1
            ST.build_ST(needCover, i)
            ST.solve1(gauss_elimination)

    length = len(gates)
    for j in range(length - 1, -1, -1):
        ans.append(gates[j])
    return ans

class topological_cnot(Optimization):
    """ optimize the cnot circuit on topological device
    https://arxiv.org/pdf/1910.14478.pdf

    use steiner tree to optimize a cnot circuit on topological device

    """
    @staticmethod
    def _run(circuit : Circuit, cnot_struct = None):
        """
        Args:
            circuit(Circuit): the circuit to be optimize
            cnot_struct(list<int>/None): the struct of cnot circuit. if None, read circuit



        """
        global TOPO, N
        topo_backward_map = read(circuit, cnot_struct)
        ans = solve()

        if len(circuit.topology) == 0:
            topo = [[True] * N] * N
        else:
            topo = [[False] * N] * N
            for topology in circuit.topology:
                topo[topology[0]][topology[1]] = topo[topology[1]][topology[0]] = True

        output = []
        for item in ans:
            GateBuilder.setGateType(GateType.CX)
            c = topo_backward_map[item.carg]
            t = topo_backward_map[item.targ]
            if topo[c][t]:
                GateBuilder.setCargs(c)
                GateBuilder.setTargs(t)
                gate = GateBuilder.getGate()
                output.append(gate)
            else:
                GateBuilder.setGateType(GateType.H)
                GateBuilder.setTargs(c)
                gate = GateBuilder.getGate()
                output.append(gate)
                GateBuilder.setTargs(t)
                gate = GateBuilder.getGate()
                output.append(gate)

                GateBuilder.setGateType(GateType.CX)
                GateBuilder.setCargs(c)
                GateBuilder.setTargs(t)
                gate = GateBuilder.getGate()
                output.append(gate)

                GateBuilder.setGateType(GateType.H)
                GateBuilder.setTargs(c)
                gate = GateBuilder.getGate()
                output.append(gate)
                GateBuilder.setTargs(t)
                gate = GateBuilder.getGate()
                output.append(gate)
        return output
