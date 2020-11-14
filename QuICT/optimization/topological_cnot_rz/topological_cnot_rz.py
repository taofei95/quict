#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 8:59 上午
# @Author  : Han Yu
# @File    : topological_cnot_rz.py

from QuICT.models import *
from .._optimization import Optimization
from queue import Queue
import numpy as np

delete_vis = []
topo = []
topo_forward_map = []
topo_backward_map = []
read_cnot = []
cnot_index = {}
th = []
input = []
stateChange = []
gates = []
ans = []

waitDeal = set()

delete_total = 0
q = 0
termNumber = 0

class Steiner_Tree(object):

    def __init__(self, n):
        global topo
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

    def buildST(self, ST_input : list, size, lower_bound):
        self.root = ST_input[size - 1]
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

    def solve0(self, gsxy : list):
        GateBuilder.setGateType(GateType.CX)
        self.solve0_dfs(self.root, gsxy)

    def solve0_dfs(self, now, gsxy : list):
        for son in self.sons[now]:
            if self.ST[son] == 0:
                GateBuilder.setCargs(now)
                GateBuilder.setTargs(son)
                gsxy[son] ^= gsxy[now]
                gate = GateBuilder.getGate()
                gates.append(gate)
            self.solve0_dfs(son, gsxy)
        if now != self.root:
            GateBuilder.setCargs(self.father[now])
            GateBuilder.setTargs(now)
            gate = GateBuilder.getGate()
            gsxy[now] ^= gsxy[self.father[now]]
            gates.append(gate)

    def solve1(self, gsxy : list):
        GateBuilder.setGateType(GateType.CX)
        self.solve1_dfs0(self.root, gsxy)
        self.solve1_dfs2(self.root, gsxy)

    def solve1_dfs0(self, now, gsxy : list):
        for son in self.sons[now]:
            self.solve1_dfs0(son, gsxy)

        if self.ST[now] != 0:
            for son in self.sons[now]:
                if self.ST[son] == 0:
                    GateBuilder.setTargs(now)
                    GateBuilder.setCargs(son)
                    gsxy[now] ^= gsxy[son]
                    gate = GateBuilder.getGate()
                    gates.append(gate)
                    self.solve1_dfs1(son, gsxy)

    def solve1_dfs1(self, now, gsxy: list):
        for son in self.sons[now]:
            if self.ST[son] == 0:
                GateBuilder.setTargs(now)
                GateBuilder.setCargs(son)
                gsxy[now] ^= gsxy[son]
                gate = GateBuilder.getGate()
                gates.append(gate)
                self.solve1_dfs1(son, gsxy)

    def solve1_dfs2(self, now, gsxy: list):
        for son in self.sons[now]:
            self.solve1_dfs2(son, gsxy)
        if now != self.root:
            GateBuilder.setTargs(self.father[now])
            GateBuilder.setCargs(now)
            gsxy[self.father[now]] ^= gsxy[now]
            gate = GateBuilder.getGate()
            gates.append(gate)

ST = Steiner_Tree(0)

def delete_dfs(now):
    global delete_vis, topo, topo_forward_map, topo_backward_map
    global delete_total, q
    delete_vis[now] = True
    for i in range(q - 1, -1, -1):
        if now != i and topo[now][i] and not delete_vis[i]:
            delete_dfs(i)
    topo_forward_map[now] = delete_total
    topo_backward_map[delete_total] = now
    delete_total += 1

def read(circuit):
    global topo, delete_vis, topo_forward_map, topo_backward_map, \
        read_cnot, cnot_index, input, th
    global waitDeal
    global q, delete_total, termNumber
    global ST
    waitDeal = set()
    q = circuit.circuit_length()
    if len(circuit.topology) == 0:
        topo = [[True] * q] * q
    else:
        topo = [[False] * q] * q
        for topology in circuit.topology:
            topo[topology[0]][topology[1]] = topo[topology[1]][topology[0]] = True
    delete_vis = [0] * q
    delete_total = 0
    topo_forward_map = [0] * q
    topo_backward_map = [0] * q
    delete_dfs(q - 1)

    termNumber = 0
    read_cnot = []
    for i in range(q):
        read_cnot.append(1 << i)

    cnot_index = dict()
    input = []
    th = []

    for i in range(len(circuit.gates)):
        gate = circuit.gates[i]
        if gate.type() == GateType.CX:
            read_cnot[topo_forward_map[gate.targ]] ^= \
                    read_cnot[topo_forward_map[gate.carg]]
        elif gate.type() == GateType.Rz:
            index = cnot_index.setdefault(read_cnot[topo_forward_map[gate.targ]], 0)
            if index != 0:
                th[index - 1] += gate.pargs
            else:
                termNumber += 1
                cnot_index[read_cnot[topo_forward_map[gate.targ]]] = termNumber
                th.append(gate.pargs)
                input.append(read_cnot[topo_forward_map[gate.targ]])
                waitDeal.add(termNumber - 1)
    ST = Steiner_Tree(q)

def solve():
    global stateChange, gates, input, ans
    global waitDeal
    global q

    ans = []
    flag = False
    firstIn = True
    stateChange = []
    for i in range(q):
        stateChange.append(1 << i)

    while len(waitDeal) > 0 or not flag:
        gates = []
        a = [0] * q
        total = 0
        gsxy = []
        needDeal = []
        if len(waitDeal) > 0:
            GateBuilder.setGateType(GateType.Rz)
            for it in waitDeal:
                val = input[it]
                for i in range(q - 1, -1, -1):
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
                    if total >= q:
                        break

            for i in range(total):
                waitDeal.remove(needDeal[i])

            for j in range(q):
                val = 1 << j
                for i in range(q - 1, -1, -1):
                    if val & (1 << i) != 0:
                        if a[i] == 0:
                            a[i] = val
                            break
                        val ^= a[i]

                if val > 0:
                    gsxy.append(1 << j)
                    total += 1
                    if total >= q:
                        break
        else:
            for i in range(q):
                gsxy.append(read_cnot[i])
                total += 1
            flag = True
        if not firstIn:
            u = []
            v = []
            tempChange = []
            for i in range(q):
                tempChange.append(stateChange[i])
            for i in range(q):
                j = q
                for t in range(i, q):
                    if (1 << i) & tempChange[t]:
                        j = t
                        break
                if j == q:
                    raise Exception("ERROR 3")
                if j != i:
                    u.append(j)
                    v.append(i)
                    tempChange[i] ^= tempChange[j]
                for j in range(q):
                    if j != i:
                        if (1 << i) & tempChange[j]:
                            u.append(i)
                            v.append(j)
                            tempChange[j] ^= tempChange[i]
            length = len(u)
            for i in range(length - 1, -1, -1):
                for j in range(q):
                    if gsxy[j] & (1 << v[i]):
                        gsxy[j] ^= 1 << u[i]
        else:
            firstIn = False

        if total == 0:
            break

        if total != q:
            raise Exception("ERROR0")

        for i in range(total):
            j = total
            for t in range(i, total):
                if gsxy[t] & (1 << i):
                    j = t
                    break

            if j >= total:
                raise Exception("ERROR1")

            bfs = Queue()
            pre = [-1] * q
            bfs.put(j)
            while not bfs.empty():
                u = bfs.get()
                if u == i:
                    break
                for j in range(q):
                    if j != u and topo[u][j]:
                        if pre[j] == -1:
                            pre[j] = u
                            if j == i:
                                break
                            bfs.put(j)
                if pre[i] != -1:
                    break

            # 寻找1
            u = pre[i]
            target = i
            GateBuilder.setGateType(GateType.CX)
            while u != -1:
                GateBuilder.setCargs(u)
                GateBuilder.setTargs(target)
                gate = GateBuilder.getGate()
                gates.append(gate)
                gsxy[target] ^= gsxy[u]
                target = u
                u = pre[u]

            # 寻找需要消去的1
            needCover = []
            cover = 0
            for j in range(i + 1, total):
                if gsxy[j] & (1 << i):
                    needCover.append(j)
                    cover += 1
            if cover > 0:
                needCover.append(i)
                cover += 1
                ST.buildST(needCover, cover, i)
                ST.solve0(gsxy)

            # 寻找回消
            fz_gsxy = [0] * (i + 1)
            xor_result = [0] * (i + 1)
            for j in range(i + 1, total):
                fz_gsxy.append(gsxy[j])
                xor_result.append(1 << j)

            for j in range(i + 1, total):
                k = total
                for t in range(j, total):
                    if fz_gsxy[t] & (1 << j):
                        k = t
                        break
                if k == total:
                    raise Exception("ERROR2")
                fz_gsxy[k], fz_gsxy[j] = fz_gsxy[j], fz_gsxy[k]
                xor_result[k], xor_result[j] = xor_result[j], xor_result[k]
                for k in range(j + 1, total):
                    if fz_gsxy[k] & (1 << j):
                        fz_gsxy[k] ^= fz_gsxy[j]
                        xor_result[k]  ^= xor_result[j]

            val = gsxy[i]
            result_xy = 0
            for j in range(i + 1, total):
                if val & (1 << j):
                    val ^= fz_gsxy[j]
                    result_xy ^= xor_result[j]

            needCover = []
            cover = 0
            for j in range(i + 1, total):
                if result_xy & (1 << j):
                    needCover.append(j)
                    cover += 1

            if cover > 0:
                needCover.append(i)
                cover += 1
                ST.buildST(needCover, cover, i)
                ST.solve1(gsxy)

        length = len(gates)
        for j in range(length - 1, -1, -1):
            ans.append(gates[j])
            if gates[j].type() == GateType.CX:
                stateChange[gates[j].targ] ^= stateChange[gates[j].carg]


class topological_cnot_rz(Optimization):
    @staticmethod
    def __run__(circuit: Circuit, *pargs):
        """
        cnot_rz电路化简
        :param circuit: 需变化电路
        :return: 返回新电路门的数组
        """
        global topo, topo_backward_map
        read(circuit)
        solve()

        if len(circuit.topology) == 0:
            topo = [[True] * q] * q
        else:
            topo = [[False] * q] * q
            for topology in circuit.topology:
                topo[topology[0]][topology[1]] = topo[topology[1]][topology[0]] = True

        output = []
        total = 0
        for item in ans:
            if item.type() == GateType.Rz or topo[topo_backward_map[item.carg]][topo_backward_map[item.targ]]:
                total += 1
            else:
                total += 5
        for item in ans:
            if item.type() == GateType.CX:
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
            else:
                GateBuilder.setGateType(GateType.Rz)
                GateBuilder.setPargs(item.pargs)
                GateBuilder.setTargs(topo_backward_map[item.targ])
                gate = GateBuilder.getGate()
                output.append(gate)
        return output
