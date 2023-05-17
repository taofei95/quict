import random

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import H, Unitary, X
from QuICT.simulation.state_vector import StateVectorSimulator

flag = 0
ballot = []
ballot_vector = []  # The vector corresponding to the ballot
v_num_i = int(input("Please input the number of voters"))
param_i = int(input("Please input the param"))
for i in range(v_num_i):
    while True:
        x = int(input("Please input your ballot (0 for yes, 1 for no):"))
        if x == 0 or x == 1:
            ballot.append(x)
            ballot_vector.append([])
            if ballot[i] == 0:
                for j in range(param_i):
                    ballot_vector[i].append(0)
            elif ballot[i] == 1:
                for j in range(param_i):
                    ballot_vector[i].append(random.randint(0, 1))
                if sum(ballot_vector[i]) == 0:
                    ballot_vector[i][random.randint(0, 1)] = 1
            break
        else:
            print("Error! you must input a valid letter!Please try again!\n")
            continue
if sum(ballot) != 0 and np.mod(sum(ballot), 2) == 0:
    flag = 1  # 偶数个投票者


class Vote:

    def __init__(self, simulator=StateVectorSimulator()):
        self.r = None  # s[i][j]: Whether the ith voter executes gate H on the j-TH single photon,
        #  where 1 indicates execution
        self.s = None  # r[i][j]: Whether the ith voter executes gate uy on the j-TH single photon,
        #  where 1 indicates execution
        self.mea_1 = []  # The results of single photon measurement are prepared
        self.mea_2 = []  # The final measurement
        self.eff = []  # Column number, the label corresponding to the single photon satisfying the valid condition
        self.simulator = simulator

    def run(self, v_num: int, param: int, q: int):
        """Run the voting algorithm

            Args:
                v_num: the number of voters
                param: A parameter related to the probability of getting the result correctly(1-1/(2^k))
                q: a parameter.the number of single photons, p_num = 2*(param + q)
        """
        if v_num != v_num_i:
            raise ValueError("v_num should equal to v_num_i")
        if param != param_i:
            raise ValueError("param should equal to param_i")
        p_num = 2 * (param + q)
        self.s = np.random.randint(0, 1, (p_num, v_num))
        self.r = np.random.randint(0, 1, (p_num, v_num))
        circuits = []
        uy = Unitary(np.array([[0, 1], [-1, 0]]))
        y = []  # ballot_vector  plus the value of the single photon used to calculate in r
        ys = []  # The sum of each column of y
        w = []  # the final of each event
        #  Single photon preparation
        for i in range(p_num):
            circuits.append(Circuit(1))
            if random.randint(0, 1) == 1:
                X | circuits[i](0)
            if random.randint(0, 1) == 1:
                H | circuits[i](0)
            self.mea_1.append(self.simulator.run(circuits[i]))
        #  Execute quantum gate
        for i in range(v_num):
            for j in range(p_num):
                if self.s[i][j] == 1:
                    H | circuits[j](0)
                if self.r[i][j] == 1:
                    uy | circuits[j](0)
        for i in range(p_num):
            self.mea_2.append(self.simulator.run(circuits[i]))
        # Eavesdropping detection
        self.detect_eva(v_num, param, q)
        #  Count the vote
        for i in range(v_num):
            y.append([])
            for j in range(param):
                y[i].append(np.mod(ballot_vector[i][j] +
                                   self.r[i][self.eff[int(p_num / 2 - param) + j]], 2))
        for col in zip(*y):  # 二维列表取列
            ys.append(np.mod(sum(col), 2))
        for j in range(param):
            if self.mea_1[int(p_num / 2 - param) + j].all() \
                    == self.mea_2[int(p_num / 2 - param) + j].all():
                w.append(ys[j])
            else:
                w.append(np.mod(ys[j] + 1, 2))
            if w[j] == 1:
                print("Event rejected")
                return w[j]
        if flag == 1 and sum(w) == 0:
            print("Event rejected,even number")
            return 1
        else:
            print("Event pass")
            return 0

    def detect_eva(self, v_num: int, param: int, q: int):
        # Check that the number of valid events is appropriate
        num = 0
        r_col_sum = []  # Record the sum of each column of r
        for col in zip(*self.s):
            if np.mod(sum(col), 2) == 0:
                self.eff.append(num)
            num += 1
        if len(self.eff) < param + q:
            print("Can't get the result")
            return
        for col in zip(*self.r):
            r_col_sum.append(np.mod(sum(col), 2))
        # Eavesdropping detection
        for i in range(q):
            if r_col_sum[self.eff[i]] == 0 and self.mea_1[self.eff[i]].all() != self.mea_2[self.eff[i]].all():
                print("Eavesdropper detected")
                break
            elif r_col_sum[self.eff[i]] == 1 and self.mea_1[self.eff[i]].all() == self.mea_2[self.eff[i]].all():
                print("Eavesdropper detected")
                break
