#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/11/5 8:46 上午
# @Author  : Han Yu
# @File    : alter_depth_decomposition.py

from .._optimization import Optimization
from QuICT.exception import CircuitStructException
from QuICT.models import *
import copy
import numpy as np

size_n = 0
cal = 0
color = []
v = []
r1 = 0
r2 = 0
gates = []


def bit(a, idx):
    return (a >> idx) & 1


def bincut(a, idx):
    return (a >> (idx + 1) << idx) + a % (1 << idx)


def bincat(a, idx, val):
    mod = 1 << idx
    return a % mod + ((a >> idx) << (idx + 1)) + val * mod


def bincaat(a, b1, b2, c1, c2):
    return bincat(bincat(a, min(b1, b2), c1 if (b1 < b2) else c2), max(b1, b2), c1 if (b1 > b2) else c2)


def binmaskcat(a1, a2, m, n):
    res = 0
    j = 0
    k = 0
    for i in range(n):
        if (m & (1 << i)) != 0:
            if (a1 & (1 << j)) != 0:
                res += (1 << i)
            j = j + 1
        else:
            if (a2 & (1 << k)) != 0:
                res += (1 << i)
            k = k + 1
    return res


class inner_perm(object):
    def __init__(self, _n=0):
        self.n = _n
        self.mp = [i for i in range(1 << _n)]
        self.ump = [i for i in range(1 << _n)]

    def make_ump(self):
        for i in range(1 << self.n):
            self.ump[self.mp[i]] = i


def copy_inner_perm(perm: inner_perm) -> inner_perm:
    newPerm = inner_perm()
    newPerm.n = perm.n
    newPerm.mp = copy.deepcopy(perm.mp)
    newPerm.ump = copy.deepcopy(perm.ump)
    return newPerm


def perm_check(p: inner_perm):
    vis = [False] * len(p.mp)
    index = 0
    for i in range(len(p.mp)):
        if vis[i]:
            continue
        vis[i] = True
        index += 1
        nxt = p.mp[i]
        while nxt != i:
            if vis[nxt]:
                raise Exception("置换错误")
            vis[nxt] = True
            nxt = p.mp[nxt]
    if index % 2 == 1:
        return False
    return True


def merge(p0: inner_perm, p1: inner_perm, rr1: int):
    if p0.n != p1.n:
        raise Exception("n不相等")
    nn = p0.n + 1
    p_list = [p0, p1, inner_perm(nn)]
    for i in range(1 << (nn - 1)):
        for j in range(2):
            p_list[2].mp[bincat(i, rr1, j)] = bincat(p_list[j].mp[i], rr1, j)
    p_list[2].make_ump()
    return p_list[2]


def subperm(p: inner_perm, rr1: int, val: int):
    n = p.n - 1
    p1 = inner_perm(n)
    for i in range(1 << n):
        t = p.mp[bincat(i, rr1, val)]
        if ((bincat(i, r1, val) ^ t) & (1 << r1)) != 0:
            raise Exception("算法出错")
        p1.mp[i] = (t >> (rr1 + 1) << rr1) + t % (1 << rr1)
    p1.make_ump()
    return p1


def inverse(p: inner_perm):
    p1 = copy_inner_perm(p)
    p1.mp, p1.ump = p1.ump, p1.mp
    return p1


def swapl(p: inner_perm, a: int, b: int):
    last_a = p.ump[a]
    last_b = p.ump[b]
    p.mp[last_a], p.mp[last_b] = p.mp[last_b], p.mp[last_a]
    p.ump[a], p.ump[b] = p.ump[b], p.ump[a]


def swapr(p: inner_perm, a: int, b: int):
    next_a = p.mp[a]
    next_b = p.mp[b]
    p.mp[a], p.mp[b] = p.mp[b], p.mp[a]
    p.ump[next_a], p.ump[next_b] = p.ump[next_b], p.ump[next_a]


def _swapr(p, a, b):
    global r1
    swapr(p, a, b)
    swapc(a, b)


def sr1(pos):
    global r1
    return pos ^ (1 << r1)


def sr2(pos):
    global r2
    return pos ^ (1 << r2)


def sr12(pos):
    return sr1(sr2(pos))


def initPerm(n) -> inner_perm:
    return inner_perm(n)


def gate2perm(gate: BasicGate, n_size):
    res = inner_perm(n_size)
    if gate.type() == GateType.Perm:
        mask = 0
        for i in range(gate.targets):
            mask += 1 << (n_size - 1 - gate.targs[i])
        for i in range(1 << (n_size - gate.targets)):
            for j in range(1 << gate.targets):
                _from = 0
                to = 0
                for k in range(gate.targets):
                    if j & (1 << k) != 0:
                        _from += 1 << (n_size - 1 - gate.targs[k])
                    if gate.pargs[j] & (1 << k) != 0:
                        to += 1 << (n_size - 1 - gate.targs[k])
                l = 0
                for k in range(size_n):
                    if mask & (1 << k) == 0:
                        if i & (1 << l) != 0:
                            _from += 1 << k
                            to += 1 << k
                        l += 1
                res.mp[_from] = to
    elif gate.type() == GateType.Swap:
        for i in range(1 << (n_size - 2)):
            temp1 = bincaat(i, n_size - 1 - gate.targs[0], n_size - 1 - gate.targs[1], 0, 1)
            temp2 = bincaat(i, n_size - 1 - gate.targs[0], n_size - 1 - gate.targs[1], 1, 0)
            res.mp[temp1], res.mp[temp2] = res.mp[temp2], res.mp[temp1]
    elif gate.type() == GateType.CX:
        for i in range(1 << (n_size - 2)):
            temp1 = bincaat(i, n_size - 1 - gate.carg, n_size - 1 - gate.targ, 1, 0)
            temp2 = bincaat(i, n_size - 1 - gate.carg, n_size - 1 - gate.targ, 1, 1)
            res.mp[temp1], res.mp[temp2] = res.mp[temp2], res.mp[temp1]
    elif gate.type() == GateType.X:
        for i in range(1 << (n_size - 1)):
            temp1 = bincat(i, n_size - 1 - gate.targ, 0)
            temp2 = bincat(i, n_size - 1 - gate.targ, 1)
            res.mp[temp1], res.mp[temp2] = res.mp[temp2], res.mp[temp1]
    elif gate.type() == GateType.ID:
        pass
    elif gate.type() == GateType.CCX:
        for i in range(1 << (n_size - 2)):
            if bincaat(i, n_size - 1 - gate.cargs[0], n_size - 1 - gate.cargs[1], 0, 0) & (
                    1 << (n_size - 1 - gate.targ)) != 0:
                temp1 = bincaat(i, n_size - 1 - gate.cargs[0], n_size - 1 - gate.cargs[1], 1, 1) ^ (
                            1 << (n_size - 1 - gate.targ))
                temp2 = bincaat(i, n_size - 1 - gate.cargs[0], n_size - 1 - gate.cargs[1], 1, 1)
                res.mp[temp1], res.mp[temp2] = res.mp[temp2], res.mp[temp1]
    else:
        raise CircuitStructException("应只包含X, CX, CCX, ID, Swap, Perm")
    res.make_ump()
    return res


def perm2gate(perm, mask, nn, datas):
    GateBuilder.setGateType(GateType.Perm)
    targs = []
    for i in range(nn):
        if (1 << i) & mask != 0:
            targs.append(nn - 1 - i)
    GateBuilder.setTargs(targs)
    params = []
    for i in range(1 << perm.n):
        params.append(datas[perm.mp[i]])
    GateBuilder.setPargs(params)
    return GateBuilder.getGate()


def comp(p1: inner_perm, p2: inner_perm):
    if p1.n != p2.n:
        raise Exception("comp错误")
    p3 = inner_perm(p1.n)
    for i in range(1 << p1.n):
        p3.mp[i] = p1.mp[p2.mp[i]]
    for i in range(1 << p1.n):
        p3.ump[i] = p2.ump[p1.ump[i]]
    return p3


def factorial(n):
    ans = 1
    for i in range(1, n):
        ans *= i
    return ans


def cantor_code(p: inner_perm):
    res = 0
    length = p.n
    for i in range(1, p.n):
        rnk = 0
        for j in range(i + 1, length):
            if p.mp[i] > p.mp[j]:
                rnk += 1
        res += rnk * factorial(length - i - 1)
    p.make_ump()
    return res


def cantor_perm(m: int, n: int):
    num = [i for i in range(1, m + 1)]
    ans = inner_perm(n)
    k = 0
    for i in range(m, 0, -1):
        r = n % factorial(i - 1)
        t = n // factorial(i - 1)
        n = r
        num.sort()
        ans.mp[k] = num[t] - 1
        k += 1
        del num[t]
    return ans


def cmp_perm(p1: inner_perm, p2: inner_perm):
    if p1.n != p2.n:
        return False
    for i in range(1 << p1.n):
        if p1.mp[i] != p2.mp[i]:
            return False
    return True


def rand_perm(n):
    p = inner_perm(n)
    p.mp = np.random.permutation(1 << n).tolist()
    return p


def mod_perm(p: inner_perm):
    global cal, v
    res = 0
    N = 1 << p.n
    v = [0] * N
    for i in range(N):
        cal += 1
        if v[i] != 0:
            continue
        v[i] = 1
        nxt = p.mp[i]
        t = 0
        while nxt != i:
            cal += 1
            v[nxt] = 1
            t ^= 1
            nxt = p.mp[nxt]
        res ^= t
    return res


def mod_perm_p(p: inner_perm):
    ans = 0
    for i in range(1 << p.n):
        if p.mp[i] > i:
            ans ^= 1
    return ans


'''
implementation of lemmas for Proposition 1
'''


def proposition_1(p: inner_perm):
    global r1, r2
    r1 = 0
    r2 = 1
    a, b = calc_ab(p, r1, r2)
    if a[3] + a[4] + b[3] + b[4] > 2 or \
            (a[3] + a[4] + b[3] + b[4] == 2 and min(b[1] + a[2], a[1] + b[2]) > 0) or \
            (a[3] + a[4] + b[3] + b[4] == 0 and (b[1] + a[2]) % 2 == 0):
        return r1, r2, lemma_1(p)
    else:
        r2 = 2
        return r1, r2, lemma_1(p)


def calc_ab(p: inner_perm, rr1, rr2):
    global r1, r2
    r1 = rr1
    r2 = rr2
    n = p.n
    a = [0] * 5
    b = [0] * 5
    for i in range(1 << (n - 2)):
        for j in range(2):
            c = [0] * 2
            for k in range(2):
                pos = bincatr1r2(i, j, k)
                if bit(p.mp[pos], r1) != bit(pos, r1):
                    c[k] = 1
                else:
                    c[k] = 0
            if j == 0:
                cc = b
            else:
                cc = a
            if c[0] == 0 and c[1] == 1:
                cc[1] += 1
            if c[0] == 1 and c[1] == 0:
                cc[2] += 1
            if c[0] == 1 and c[1] == 1:
                cc[3] += 1
            if c[0] == 0 and c[1] == 0:
                cc[4] += 1
    return a, b


def swapc(a, b):
    global color, r1
    color[a], color[b] = color[b], color[a]
    if ((a ^ b) & (1 << r1)) != 0:
        color[a] ^= 1
        color[b] ^= 1


def checkump(p: inner_perm):
    for i in range(1 << p.n):
        if p.mp[p.ump[i]] != i:
            return 0
    return 1


def bincatr1r2(a, v1, v2):
    global r1, r2
    if r1 > r2:
        sec = 1
    else:
        sec = 0
    return bincat(bincat(a, r1 - sec, v1), r2, v2)


def bincatr1(a, b):
    global r1
    return bincat(a, r1, b)


'''
 Output [p1, p2, p3]: SC(r1), SC(r2), SC(r1) such that comp(p, p1, p2, p3) in A(r1), if
    a3 + a4 + b4 + b4 > 2 holds or,
    a3 + a4 + b3 + b4 = 2 and min{b1 + a2, a1 + b2} > 0 hold or,
    a3 + a4 + b3 + b4 = 0 and b1 + a2 is even (equivalently a1 + b2 is even) hold.

   NOTE:
    a = (a1 - a2 + b2 - b1) / 2
    b = b1 + a2
    c = (a3 + a4 + b3 + b4) / 2
'''


def lemma_1(p: inner_perm):
    global r1, r2
    global color
    n = p.n
    p1 = lemma_2(p)[0]
    a = 0
    b = 0
    c = 0
    st = [[], [], [], []]
    p2 = inner_perm(n)
    p3 = inner_perm(n)
    tmp_ptr = comp(p, p1)
    for i in range(1 << (n - 2)):
        t1 = 0
        t2 = 0
        for j in range(2):
            for k in range(2):
                pos = bincatr1r2(i, j, k)
                if ((pos ^ tmp_ptr.mp[pos]) & (1 << r1)) != 0:
                    color[pos] = 1
                else:
                    color[pos] = 0
                t1 += color[pos]
                if k == 0:
                    t2 ^= color[pos]
                else:
                    t2 ^= 0
        if t1 == 0:
            c += 1
            st[3].append(i)
        elif t2 == 1:
            a += 1
            st[1].append(i)
        else:
            b += 1
            st[2].append(i)
    type = 0
    if a % 2 == 0 and b % 2 == 0:
        type = 1
    elif c >= 2:
        type = 2
    elif c == 1 and b > 0 and a + b > 2:
        type = 3
    if type == 0:
        return []

    while (type == 3 and a + b >= 5 and a >= 3) or (type != 3 and a >= 2):
        a -= 2
        pos0 = bincatr1r2(st[1].pop(), 0, 0)
        pos1 = bincatr1r2(st[1].pop(), 0, 0)

        _swapr(p2, pos0, sr2(pos1))
        _swapr(p2, sr1(pos0), sr12(pos1))

        if color[pos0]:
            pos0, pos1 = pos1, pos0
        _swapr(p3, sr1(pos0), pos1)
        _swapr(p3, sr12(pos0), sr2(pos1))

    while (type == 3 and a + b >= 5) or (type != 3 and b >= 2):
        b -= 2
        pos0 = bincatr1r2(st[2].pop(), 0, 0)
        pos1 = bincatr1r2(st[2].pop(), 0, 0)

        _swapr(p2, pos0, sr2(pos1))
        _swapr(p2, sr1(pos0), sr12(pos1))

        if color[pos1]:
            pos0, pos1 = pos1, pos0
        _swapr(p3, pos0, sr1(pos0))
        _swapr(p3, sr2(pos0), sr12(pos0))

    # Case #1
    if a == 1 and b == 1 and c >= 2:
        pos1 = bincatr1r2(st[1].pop(), 0, 0)
        pos2 = bincatr1r2(st[2].pop(), 0, 0)
        pos31 = bincatr1r2(st[3].pop(), 0, 0)
        pos32 = bincatr1r2(st[3].pop(), 0, 0)

        if color[pos2] != 0:
            t = 1 << r2
        else:
            t = 0
        if color[pos1] == 0:
            t ^= 1 << r1
        else:
            t ^= 0
        pos1 ^= t
        pos2 ^= t
        pos31 ^= t
        pos32 ^= t
        _swapr(p1, sr1(pos1), sr1(pos32))
        _swapr(p1, sr12(pos1), sr12(pos32))

        _swapr(p1, pos2, pos31)
        _swapr(p1, sr2(pos2), sr2(pos31))
        _swapr(p1, pos1, pos31)
        _swapr(p1, sr2(pos1), sr2(pos31))

        _swapr(p2, pos1, pos31)
        _swapr(p2, sr1(pos1), sr1(pos31))
        _swapr(p2, sr2(pos2), pos32)
        _swapr(p2, sr12(pos2), sr1(pos32))

        _swapr(p3, pos1, sr1(pos32))
        _swapr(p3, sr2(pos1), sr12(pos32))
    elif a == 1 and b == 0 and c >= 2:
        pos1 = bincatr1r2(st[1].pop(), 0, 0)
        pos31 = bincatr1r2(st[3].pop(), 0, 0)
        pos32 = bincatr1r2(st[3].pop(), 0, 0)

        if color[pos1] == 0:
            t = 1 << r2
        else:
            t = 0
        pos1 ^= t
        pos31 ^= t
        pos32 ^= t

        _swapr(p1, pos1, sr1(pos32))
        _swapr(p1, sr2(pos1), sr12(pos32))
        _swapr(p1, sr1(pos31), pos32)
        _swapr(p1, sr12(pos31), sr2(pos32))

        _swapr(p2, pos1, sr2(pos32))
        _swapr(p2, sr1(pos1), sr12(pos32))

        _swapr(p3, pos1, sr1(pos1))
        _swapr(p3, sr2(pos1), sr12(pos1))
        _swapr(p3, sr1(pos31), pos32)
        _swapr(p3, sr12(pos31), sr2(pos32))
    elif a == 0 and b == 1 and c >= 2:
        pos2 = bincatr1r2(st[2].pop(), 0, 0)
        pos31 = bincatr1r2(st[3].pop(), 0, 0)
        pos32 = bincatr1r2(st[3].pop(), 0, 0)
        if color[pos2] != 0:
            t = 1 << r2
        else:
            t = 0
        pos2 ^= t
        pos31 ^= t
        pos32 ^= t

        _swapr(p1, pos2, sr1(pos32))
        _swapr(p1, sr2(pos2), sr12(pos32))
        _swapr(p1, sr1(pos31), pos32)
        _swapr(p1, sr12(pos31), sr2(pos32))

        _swapr(p2, pos2, pos32)
        _swapr(p2, sr1(pos2), sr1(pos32))

        _swapr(p3, pos2, sr1(pos2))
        _swapr(p3, sr2(pos2), sr12(pos2))
        _swapr(p3, sr1(pos31), pos32)
        _swapr(p3, sr12(pos31), sr2(pos32))

    elif a == 2 and b == 1 and c >= 1:
        pos11 = bincatr1r2(st[1].pop(), 0, 0)
        pos12 = bincatr1r2(st[1].pop(), 0, 0)
        pos2 = bincatr1r2(st[2].pop(), 0, 0)
        pos3 = bincatr1r2(st[3].pop(), 0, 0)

        if color[pos2] != 0:
            t = 1 << r2
        else:
            t = 0
        if color[pos11] == 0:
            t ^= 1 << r1
        else:
            t ^= 0
        pos11 ^= t
        pos12 ^= t
        pos2 ^= t
        pos3 ^= t

        _swapr(p1, pos3, pos2)
        _swapr(p1, sr2(pos3), sr2(pos2))
        _swapr(p1, sr1(pos3), pos3)
        _swapr(p1, sr12(pos3), sr2(pos3))
        _swapr(p1, sr1(pos11), sr1(pos3))
        _swapr(p1, sr12(pos11), sr12(pos3))

        _swapr(p2, pos11, sr2(pos2))
        _swapr(p2, sr1(pos11), sr12(pos2))
        _swapr(p2, sr2(pos11), sr2(pos12))
        _swapr(p2, sr12(pos11), sr12(pos12))
        _swapr(p2, sr2(pos3), pos2)
        _swapr(p2, sr12(pos3), sr1(pos2))
        _swapr(p2, pos12, sr2(pos3))
        _swapr(p2, sr1(pos12), sr12(pos3))

        _swapr(p3, sr1(pos11), pos2)
        _swapr(p3, sr12(pos11), sr2(pos2))
        _swapr(p3, sr1(pos2), pos3)
        _swapr(p3, sr12(pos2), sr2(pos3))
    elif a == 0 and b == 3 and c >= 1:
        pos21 = bincatr1r2(st[2].pop(), 0, 0)
        pos22 = bincatr1r2(st[2].pop(), 0, 0)
        pos23 = bincatr1r2(st[2].pop(), 0, 0)
        pos3 = bincatr1r2(st[3].pop(), 0, 0)

        if color[pos21] != 0:
            t = 1 << r2
        else:
            t = 0

        pos21 ^= t
        pos22 ^= t
        pos23 ^= t
        pos3 ^= t

        _swapr(p1, pos21, sr1(pos21))
        _swapr(p1, sr2(pos21), sr12(pos21))
        _swapr(p1, sr1(pos3), pos3)
        _swapr(p1, sr12(pos3), sr2(pos3))
        _swapr(p1, sr1(pos22), sr1(pos3))
        _swapr(p1, sr12(pos22), sr12(pos3))
        _swapr(p1, pos23, sr1(pos22))
        _swapr(p1, sr2(pos23), sr12(pos22))

        _swapr(p2, pos23, pos22)
        _swapr(p2, sr1(pos23), sr1(pos22))
        _swapr(p2, pos21, pos23)
        _swapr(p2, sr1(pos21), sr1(pos23))
        _swapr(p2, sr2(pos3), sr2(pos23))
        _swapr(p2, sr12(pos3), sr12(pos23))
        _swapr(p2, sr2(pos22), sr2(pos3))
        _swapr(p2, sr12(pos22), sr12(pos3))
        _swapr(p2, sr2(pos21), sr2(pos22))
        _swapr(p2, sr12(pos21), sr12(pos22))

        _swapr(p3, sr1(pos21), pos23)
        _swapr(p3, sr12(pos21), sr2(pos23))
        _swapr(p3, sr1(pos23), pos3)
        _swapr(p3, sr12(pos23), sr2(pos3))
    elif a == 1 and b == 2 and c >= 1:
        pos1 = bincatr1r2(st[1].pop(), 0, 0)
        pos21 = bincatr1r2(st[2].pop(), 0, 0)
        pos22 = bincatr1r2(st[2].pop(), 0, 0)
        pos3 = bincatr1r2(st[3].pop(), 0, 0)

        if color[pos21] != 0:
            t = 1 << r2
        else:
            t = 0
        if color[pos1] == 0:
            t ^= 1 << r1
        else:
            t ^= 0
        pos1 ^= t
        pos21 ^= t
        pos22 ^= t
        pos3 ^= t

        _swapr(p1, sr1(pos3), pos3)
        _swapr(p1, sr12(pos3), sr2(pos3))
        _swapr(p1, sr1(pos21), sr1(pos3))
        _swapr(p1, sr12(pos21), sr12(pos3))
        _swapr(p1, pos21, sr1(pos21))
        _swapr(p1, sr2(pos21), sr12(pos21))

        _swapr(p2, pos22, pos21)
        _swapr(p2, sr1(pos22), sr1(pos21))
        _swapr(p2, sr2(pos3), pos22)
        _swapr(p2, sr12(pos3), sr1(pos22))
        _swapr(p2, pos1, sr2(pos3))
        _swapr(p2, sr1(pos1), sr12(pos3))

        _swapr(p3, sr1(pos1), pos22)
        _swapr(p3, sr12(pos1), sr2(pos22))
        _swapr(p3, sr1(pos22), pos3)
        _swapr(p3, sr12(pos22), sr2(pos3))
    elif a != 0 and b != 0:
        raise Exception("Error")

    ans = [p1, p2, p3]
    return ans


def lemma_2(p) -> list:
    global r1, r2, color
    a = [0] * 5
    b = [0] * 5
    st = [[], []]
    for i in range(5):
        st[0].append([])
        st[1].append([])
    n = p.n
    p1 = inner_perm(n)
    for i in range(1 << (n - 2)):
        for j in range(2):
            c = [0] * 2
            for k in range(2):
                pos = bincatr1r2(i, j, k)
                if bit(p.mp[pos], r1) != bit(pos, r1):
                    c[k] = color[pos] = 1
                else:
                    c[k] = color[pos] = 0
            if j == 0:
                cc = a
            else:
                cc = b
            if c[0] == 0 and c[1] == 1:
                cc[1] += 1
                st[j][1].append(i)
            if c[0] == 1 and c[1] == 0:
                cc[2] += 1
                st[j][2].append(i)
            if c[0] == 1 and c[1] == 1:
                cc[3] += 1
                st[j][3].append(i)
            if c[0] == 0 and c[1] == 0:
                cc[4] += 1
                st[j][4].append(i)
    while a[3] > 0 or b[3] > 0:
        pos0 = pos1 = 0
        if a[3] > 0:
            if b[3] > 0:
                lp0 = st[0][3].pop()
                lp1 = st[1][3].pop()
                pos0 = bincatr1r2(lp0, 0, 0)
                pos1 = bincatr1r2(lp1, 1, 0)
                a[3] -= 1
                b[3] -= 1
                a[4] += 1
                b[4] += 1
                color[pos0] = color[sr2(pos0)] = color[pos1] = color[sr2(pos1)] = 0
                st[0][4].append(lp0)
                st[1][4].append(lp1)
            elif b[1] > 0:
                lp0 = st[0][3].pop()
                lp1 = st[1][1].pop()
                pos0 = bincatr1r2(lp0, 0, 0)
                pos1 = bincatr1r2(lp1, 1, 0)
                a[3] -= 1
                b[1] -= 1
                a[2] += 1
                b[4] += 1
                st[0][2].append(lp0)
                st[1][4].append(lp1)
            elif b[2] > 0:
                lp0 = st[0][3].pop()
                lp1 = st[1][2].pop()
                pos0 = bincatr1r2(lp0, 0, 0)
                pos1 = bincatr1r2(lp1, 1, 0)
                a[3] -= 1
                b[2] -= 1
                a[1] += 1
                b[4] += 1
                st[0][1].append(lp0), st[1][4].append(lp1)

        else:
            if a[1] > 0:
                lp1 = st[1][3].pop()
                lp0 = st[0][1].pop()
                pos1 = bincatr1r2(lp1, 1, 0)
                pos0 = bincatr1r2(lp0, 0, 0)
                b[3] -= 1
                a[1] -= 1
                b[2] += 1
                a[4] += 1
                st[0][4].append(lp0)
                st[1][2].append(lp1)
            elif a[2] > 0:
                lp1 = st[1][3].pop()
                lp0 = st[0][2].pop()
                pos1 = bincatr1r2(lp1, 1, 0)
                pos0 = bincatr1r2(lp0, 0, 0)
                b[3] -= 1
                a[2] -= 1
                b[1] += 1
                a[4] += 1
                st[0][4].append(lp0)
                st[1][1].append(lp1)
        _swapr(p1, pos0, pos1)
        _swapr(p1, sr2(pos0), sr2(pos1))

    sw = 2
    while a[sw] > 0 and b[sw] > 0:
        lp1 = st[1][sw].pop()
        pos1 = bincatr1r2(lp1, 1, 0)
        lp0 = st[0][sw].pop()
        pos0 = bincatr1r2(lp0, 0, 0)
        b[sw] -= 1
        a[sw] -= 1
        b[3 - sw] += 1
        a[3 - sw] += 1
        st[0][3 - sw].append(lp0), st[1][3 - sw].append(lp1)
        _swapr(p1, pos0, pos1)
        _swapr(p1, sr2(pos0), sr2(pos1))

    for i in range(1 << (n - 2)):
        for j in range(2):
            pos = bincatr1r2(i, j, 0)
            if j == 0:
                ab = a
            else:
                ab = b
            while len(st[j][1]) != 0 and st[j][1][-1] < i:
                st[j][1].pop()
                ab[1] -= 1
            while len(st[j][2]) != 0 and st[j][2][-1] < i:
                st[j][2].pop()
                ab[2] -= 1
            if ab[1] > 0:
                if color[pos] == 0 and color[sr2(pos)] == 1:
                    continue
                elif color[pos] == 1 and color[sr2(pos)] == 0:
                    st[j][2].append(st[j][1][-1])
                    ab[2] += 1
                _swapr(p1, pos, bincatr1r2(st[j][1][-1], j, 0))
                _swapr(p1, sr2(pos), bincatr1r2(st[j][1][-1], j, 1))
                st[j][1].pop()
                ab[1] -= 1
            elif ab[2] > 0:
                if color[pos] == 1 and color[sr2(pos)] == 0:
                    continue
                _swapr(p1, pos, bincatr1r2(st[j][2][-1], j, 0))
                _swapr(p1, sr2(pos), bincatr1r2(st[j][2][-1], j, 1))
                st[j][2].pop()
                ab[2] -= 1

    return [p1]


'''
implementation of lemmas for Proposition 2
'''


def lemma_6(p: inner_perm, rr1):
    global v
    n = p.n
    ans = [inner_perm(n)]
    st = []
    for i in range(6):
        st.append([])
    while True:
        flg = 0
        now_ptr = comp(p, ans[0])
        v = [0] * (1 << n)
        cir = []
        for i in range(1 << n):
            cir = []
            if v[i] != 0:
                continue
            v[i] = 1
            cir.append(i)
            now = now_ptr.mp[i]
            while now != i:
                v[now] = 1
                cir.append(now)
                now = now_ptr.mp[now]
            if len(cir) == 3 or len(cir) == 5:
                flg = 1
                break

        if flg == 0:
            break
        exc = set()
        for i in cir:
            exc.add(i)
        op = sr1(cir[0])
        exc.add(op)
        prev = op
        nxt = op
        for i in range(5):
            nxt = p.mp[nxt]
            prev = p.ump[prev]
            exc.add(nxt)
            exc.add(prev)
        for i in range(1 << (n - 1) + 1):
            if i == 1 << (n - 1):
                raise Exception("error!")
            if bincatr1(i, 0) in exc or bincatr1(i, 1) in exc:
                continue
            swapr(ans[0], cir[0], bincatr1(i, bit(cir[0], rr1)))
            swapr(ans[0], op, bincatr1(i, bit(op, rr1)))
            break
    return ans


def proposition_2(p: inner_perm, rr1) -> list:
    global v
    n = p.n
    p0 = subperm(p, rr1, 0)
    p1 = subperm(p, rr1, 1)
    tmp_ptr = comp(inverse(p0), p1)
    v = [0] * (1 << (n - 1))
    for i in range(1 << (n - 1)):
        if v[i] != 0:
            continue
        nxt = tmp_ptr.mp[i]
        num = 1
        while nxt != i:
            v[nxt] = 1
            num += 1
            nxt = tmp_ptr.mp[nxt]
    tmp_ptr = lemma_6(tmp_ptr, rr1)[0]
    ans = [merge(inner_perm(n - 1), tmp_ptr, rr1)]
    tmp_ptr = comp(comp(inverse(p0), p1), tmp_ptr)

    vp = proposition_3(tmp_ptr, 1, 0)
    ans.append(merge(inverse(vp[0]), inverse(vp[0]), rr1))
    ans.append(merge(inner_perm(n - 1), inverse(vp[2]), rr1))
    ans.append(merge(inner_perm(n - 1), inverse(vp[1]), rr1))
    ans.append(merge(comp(vp[0], inverse(p0)), comp(vp[0], inverse(p0)), rr1))

    return ans


def theorem_1(p):
    global v

    if p.n < 6:
        raise Exception("应给出位数大于等于6的电路")

    if not perm_check(p):
        # raise Exception("应给出偶置换")
        return [-1] * 5, []

    v = [0] * (1 << p.n)
    for i in range(1 << p.n):
        v[p.mp[i]] = 1
    rr1, rr2, ans = proposition_1(p)
    vp2 = proposition_2(comp(comp(comp(p, ans[0]), ans[1]), ans[2]), r1)
    if rr1 == 0:
        r3 = 1
    else:
        r3 = 0
    r4 = 3 - rr1 - r3
    vp2[0] = comp(ans[2], vp2[0])
    ans.pop()
    ans.extend(vp2)
    return [0, rr1, rr2, r3, r4], ans


def proposition_3(p, rr1, rr2):
    global v
    n = p.n
    ans = [inner_perm(n), inner_perm(n), inner_perm(n)]
    st = [[], []]

    cnum = 0
    v = [0] * (1 << n)
    for i in range(1 << n):
        if v[i] != 0:
            continue
        v[i] = 1
        vp = [i]
        now = i
        nxt = p.mp[now]
        while v[nxt] == 0:
            now = nxt
            v[now] = 1
            vp.append(now)
            nxt = p.mp[now]
        st[len(vp) % 2].append(vp)
        cnum += 1

    pos = -1
    post = 0
    bg = 0
    while len(st[0]) != 0 or len(st[1]) != 0:
        if len(st[0]) == 0:
            t = 1
        else:
            t = 0
        c0 = st[t].pop()
        c1 = st[t].pop()
        if len(c0) > len(c1):
            c0, c1 = c1, c0
        if (len(c0) + len(c1)) % 4 == 0:
            vp = rpack(c0, c1, ans[0], bg, rr1, rr2)
            bg += (len(c0) + len(c1)) // 4
        else:
            if post == 0:
                _bg = bg + 1
                _post = bg
            else:
                _bg = bg
                _post = pos
            vp = tpack(c0, c1, ans[0], _bg, _post, post, rr1, rr2)
            post ^= 1
            pos = bg
            bg += (len(c0) + len(c1)) // 4 + post

        ans[1] = comp(ans[1], vp[0])
        ans[2] = comp(ans[2], vp[1])

    ans[0].make_ump()
    ans[0] = inverse(ans[0])
    return ans


def rpack(c0, c1, ans0, bg, rr1, rr2):
    return tpack(c0, c1, ans0, bg, -1, -1, rr1, rr2)


def make_circle(c, p: inner_perm):
    for i in range(len(c)):
        if i + 1 < len(c):
            p.mp[c[i]] = c[i + 1]
        else:
            p.mp[c[i]] = c[0]


def make_circle_2(c, p: inner_perm, bg, n, pos, post, rr1, rr2):
    cc = []
    for i in range(len(c)):
        if rr2 > rr1:
            sec = 1
        else:
            sec = 0
        j = bincut(bincut(i, rr1), rr2 - sec)
        if (bg + n > j >= bg) or (j == pos and bit(i, rr2) == post):
            cc.append(c[i])
    for i in range(len(cc) - 1):
        if 1 < len(c):
            sec = cc[i + 1]
        else:
            sec = c[0]
        p.mp[cc[i]] = i + sec


def tpack(c0, c1, ans0, bg, pos, post, rr1, rr2):
    global r1, r2
    r1 = rr1
    r2 = rr2
    ans = [inner_perm(ans0.n), inner_perm(ans0.n)]
    a = len(c0)
    b = len(c1)
    k = a // 2
    l = b // 2
    if a == b:
        n = (a + b) // 4
        for j in range(2):
            cir = []
            for i in range(n):
                cir.append(bincatr1r2(i + bg, j, 1))
            if pos + 1 != 0:
                cir.append(bincatr1r2(pos, j, post))
            for i in range(n - 1, -1, -1):
                cir.append(bincatr1r2(i + bg, j, 0))
            # print("--------")
            # print(cir)
            # print(ans[0].mp)
            make_circle(cir, ans[0])
            # print(ans[0].mp)
            # print("--------")
            ans[0].make_ump()
            ans[1].make_ump()
            if pos + 1 != 0:
                now = bincatr1r2(pos, j, post)
            else:
                now = bincatr1r2(bg, j, 0)
            for i in range(a):
                if j != 0:
                    ans0.mp[now] = c1[i]
                else:
                    ans0.mp[now] = c0[i]
                now = ans[0].mp[now]
    elif a % 2 == 0:
        n = (a + b) // 4
        for j in range(2):
            cir = []
            for i in range(n):
                cir.append(bincatr1r2(i + bg, j, 0))
            if pos + 1 != 0:
                cir.append(bincatr1r2(pos, j, post))
            for i in range(n - 1, k - 1, -1):
                cir.append(bincatr1r2(i + bg, j, 1))
            make_circle(cir, ans[0])
            cir = []
            for i in range(k):
                cir.append(bincatr1r2(i + bg, j, 1))
            make_circle(cir, ans[0])

            cir = [bincatr1r2(bg, 0, j), bincatr1r2(bg, 1, j)]
            make_circle(cir, ans[1])
        ans[0].make_ump()
        ans[1].make_ump()
        comp_ptr = comp(ans[0], ans[1])
        now = bincatr1r2(bg, 0, 1)
        for i in range(a):
            ans0.mp[now] = c0[i]
            now = comp_ptr.mp[now]
        now = bincatr1r2(bg, 0, 0)
        for i in range(b):
            ans0.mp[now] = c1[i]
            now = comp_ptr.mp[now]
    elif a == 1 and b >= 7:
        n = (a + b) // 4
        for j in range(2):
            cir = []
            for i in range(n):
                cir.append(bincatr1r2(i + bg, j, 1))
            if pos + 1 != 0:
                cir.append(bincatr1r2(pos, j, post))
            for i in range(n - 1, -1, -1):
                cir.append(bincatr1r2(i + bg, j, 0))
            make_circle(cir, ans[0])

            cir = [bincatr1r2(bg, 0, j), bincatr1r2(bg + 1, 0, j), bincatr1r2(bg, 1, j)]
            make_circle(cir, ans[1])

        ans[0].make_ump()
        ans[1].make_ump()
        comp_ptr = comp(ans[0], ans[1])
        ans0.mp[bincatr1r2(bg, 0, 0)] = c0[0]

        now = bincatr1r2(bg, 1, 1)
        for i in range(b):
            ans0.mp[now] = c1[i]
            now = comp_ptr.mp[now]
    elif a % 2 == 1 and a >= 5 and b >= 5:
        k, l = l, k
        n1 = (k - l + 1) // 2
        n2 = l - 2
        for j in range(2):
            cir = [bincatr1r2(bg, j, 0)]
            for i in range(1, n1 + 2):
                cir.append(bincatr1r2(i + bg, j, 1))
            if pos + 1 != 0:
                cir.append(bincatr1r2(n1 + 2 - 1 + bg, j, 0))
            for i in range(n1 + 2 - 2, 0, -1):
                cir.append(bincatr1r2(i + bg, j, 0))
            if pos + 1 == 0:
                cir.append(bincatr1r2(n1 + 2 - 1 + bg, j, 0))
            for i in range(n1 + 2, n1 + n2 + 2):
                cir.append(bincatr1r2(i + bg, j, 0))
            if pos + 1 != 0:
                cir.append(bincatr1r2(pos, j, post))
            for i in range(n1 + n2 + 2 - 1, n1 + 2 - 1, -1):
                cir.append(bincatr1r2(i + bg, j, 1))
            cir.append(bincatr1r2(bg, j, 1))
            make_circle(cir, ans[0])

            cir = [bincatr1r2(bg, 1, j), bincatr1r2(1 + bg, 0, j)]
            make_circle(cir, ans[1])

        ans[0].make_ump()
        ans[1].make_ump()
        comp_ptr = comp(ans[0], ans[1])
        now = bincatr1r2(bg, 0, 0)

        for i in range(a):
            ans0.mp[now] = c0[i]
            now = comp_ptr.mp[now]
        now = bincatr1r2(bg + 1, 1, 1)
        for i in range(b):
            ans0.mp[now] = c1[i]
            now = comp_ptr.mp[now]
    else:
        raise Exception("ERROR!!")
    return ans


def ifid(inperm):
    for i in range(1 << inperm.n):
        if inperm.mp[i] != i:
            return 0
    return 1


def solve(circuit):
    global size_n, color, v, gates
    size_n = circuit.circuit_length()
    color = [0] * (1 << size_n)
    v = [0] * (1 << size_n)
    gate = initPerm(size_n)
    for every_gate in circuit.gates:
        now = gate2perm(every_gate, size_n)
        gate = comp(now, gate)
    rc = [2, 1, 2, 1, 3, 4, 1]
    r, perms = theorem_1(gate)

    # 如果不是偶置换，返回原本的门
    if r[0] == -1:
        print("注意，此处给出了偶置换，返回原本给出的门")
        gates = circuit.gates
        return

    size = 0
    for i in range(7):
        size += 1 - ifid(perms[i])

    datas = []
    for i in range(1 << (size_n - 1)):
        nowi = 0
        for j in range(size_n - 1):
            if (1 << j) & i != 0:
                nowi += 1 << (size_n - 1 - 1 - j)
        datas.append(nowi)
    datas = [i for i in range(1 << (size_n - 1))]

    for i in range(7):
        if ifid(perms[i]) == 0:
            idx = r[rc[i]]
            gperm = inner_perm(size_n - 1)
            mask = (1 << size_n) - 1 - (1 << idx)
            for k in range(1 << (size_n - 1)):
                gperm.mp[k] = bincut(perms[i].ump[bincat(k, idx, 0)], idx)
            gperm.make_ump()
            gates.append(perm2gate(gperm, mask, size_n, datas))


class alter_depth_decomposition(Optimization):
    @staticmethod
    def _run(circuit: Circuit, *pargs):
        """
        任意经典电路化简
        :param circuit: 需变化电路
        :return: 返回新电路门的数组
        """
        global gates
        gates = []
        solve(circuit)
        return gates
