import random

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import H, Unitary, X
from QuICT.simulation.state_vector import StateVectorSimulator


def v(m, k, t):
    # m voters，k is related to the probability of getting a successful result,t = 2(k+q)
    q = int(t / 2 - k)
    x = []  # ballot
    xx = []  # The sequence 0,1 corresponding to the ballot
    s = []  # s[i][j] indicates whether the j-th voter executes H gate on the i-th single photon
    r = []  # r[i][j]indicates whether the j-th voter executes UY gate on the i-th single photon
    uy = Unitary(np.array([[0, 1], [-1, 0]]))
    circuits = []
    mea_1 = []  # The initial state of a single photon
    mea_2 = []  # The final state of a single photon
    eff = []  # Valid event label有效事件对应的单光子标号
    y = []  # Record the intermediate results, equivalent to
    ys = []  # Record the intermediate results, equivalent to
    w = []  # 存储最后计算结果
    flag = 0  # 1表示有偶数个反对票
    rsum = []
    simulator = StateVectorSimulator(
        device="CPU",
        precision="double"
    )
    #  输入投票信息
    for i in range(m):
        while True:
            ballot = int(input("Please input your ballot (0 for yes, 1 for no):"))
            if ballot == 0 or ballot == 1:
                x.append(ballot)
                xx.append([])
                if x[i] == 0:
                    for j in range(k):
                        xx[i].append(0)
                elif x[i] == 1:
                    for j in range(k):
                        xx[i].append(random.randint(0, 1))
                    if sum(xx[i]) == 0:
                        xx[i][random.randint(0, k)] = 1
                break
            else:
                print("Error! you must input a valid letter!Please try again!\n")
                continue
    if sum(x) != 0 and np.mod(sum(x), 2) == 0:
        flag = 1  # 偶数个投票者
    # 制备t个单光子
    for i in range(t):
        circuits.append(Circuit(1))
        if random.randint(0, 1) == 1:
            X | circuits[i](0)
        if random.randint(0, 1) == 1:
            H | circuits[i](0)
        mea_1.append(simulator.run(circuits[i]))
    #  执行量子门
    for i in range(m):
        s.append([])
        r.append([])
        for j in range(t):
            s[i].append(random.randint(0, 1))
            r[i].append(random.randint(0, 1))
            if s[i][j] == 1:
                H | circuits[j](0)
            if r[i][j] == 1:
                uy | circuits[j](0)
    for i in range(t):
        mea_2.append(simulator.run(circuits[i]))
    num = 0
    for col in zip(*s):  # 二维列表取列
        if np.mod(sum(col), 2) == 0:
            eff.append(num)  # 列号，满足有效条件的单光子对应的标号
        num += 1
    if len(eff) < k + q:
        print("无法得到结果")
        return
    #  Eavesdropping detection
    for col in zip(*r):
        rsum.append(np.mod(sum(col), 2))
    for i in range(q):
        if rsum[eff[i]] == 0 and mea_1[eff[i]].all() != mea_2[eff[i]].all():
            print("Eavesdropper detected")
            break
        elif rsum[eff[i]] == 1 and mea_1[eff[i]].all() == mea_2[eff[i]].all():
            print("Eavesdropper detected")
            break
    #  Count the vote
    for i in range(m):
        y.append([])
        for j in range(k):
            y[i].append(np.mod(xx[i][j] + r[i][eff[q+j]], 2))
    for col in zip(*y):  # 二维列表取列
        ys.append(np.mod(sum(col), 2))
    for j in range(k):
        if mea_1[q + j].all() == mea_2[q + j].all():
            w.append(ys[j])
        else:
            w.append(np.mod(ys[j] + 1, 2))
        if w[j] == 1:
            print("事件被否决")
            return w[j]
    if flag == 1 and sum(w) == 0:
        print("事件被否决,偶数")
        return 1
    else:
        print("事件同意")
        return 0


if __name__ == "__main__":
    re = v(3, 3, 10)
    print(re)
