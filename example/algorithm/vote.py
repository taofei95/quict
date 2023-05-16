import random

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import H, Unitary, X
from QuICT.simulation.state_vector import StateVectorSimulator


class Vote:

    def __init__(self, v_num: int, param: int, p_num: int, simulator=StateVectorSimulator()):
        """Enter voting information

        Args:
            v_num: the number of voters
            param: A parameter related to the probability of getting the result correctly
            p_num: the number of single photons
        """
        self.v_num = v_num
        self.param = param
        self.p_num = p_num
        self.flag = 0
        self.ballot = []
        self.ballot_vector = []
        self.s = np.random.randint(0, 1, (p_num, v_num))  # s[i][j]: Whether the ith voter executes gate H on the j-TH
        # single photon, where 1 indicates execution
        self.r = np.random.randint(0, 1, (p_num, v_num))  # r[i][j]: Whether the ith voter executes gate uy on the j-TH
        # single photon, where 1 indicates execution
        self.mea_1 = None
        self.mea_2 = None
        self.eff = None
        for i in range(v_num):
            while True:
                x = int(input("Please input your ballot (0 for yes, 1 for no):"))
                if x == 0 or x == 1:
                    self.ballot.append(x)
                    self.ballot_vector.append([])
                    if self.ballot[i] == 0:
                        for j in range(self.param):
                            self.ballot_vector[i].append(0)
                    elif self.ballot[i] == 1:
                        for j in range(self.param):
                            self.ballot_vector[i].append(random.randint(0, 1))
                        if sum(self.ballot_vector[i]) == 0:
                            self.ballot_vector[i][random.randint(0, self.param)] = 1
                    break
                else:
                    print("Error! you must input a valid letter!Please try again!\n")
                    continue
        if sum(self.ballot) != 0 and np.mod(sum(self.ballot), 2) == 0:
            self.flag = 1  # 偶数个投票者
        self.simulator = simulator

    def circuit(self):
        """Construct Circuit"""
        cirs = []  #
        self.mea_1 = []  # The results of single photon measurement are prepared
        self.mea_2 = []  # The final measurement
        y = []
        ys = []
        w = []  # A vote on each event
        uy = Unitary(np.array([[0, 1], [-1, 0]]))
        # Step 1: Single photon preparation
        for i in range(self.p_num):
            cirs.append(Circuit(1))
            if random.randint(0, 1) == 1:
                X | cirs[i](0)
            if random.randint(0, 1) == 1:
                H | cirs[i](0)
            self.mea_1.append(self.simulator.run(cirs[i]))
        #  Execute quantum gate
        for i in range(self.v_num):
            for j in range(self.p_num):
                if self.s[i][j] == 1:
                    H | cirs[j](0)
                if self.r[i][j] == 1:
                    uy | cirs[j](0)
        for i in range(self.p_num):
            self.mea_2.append(self.simulator.run(cirs[i]))
        # Eavesdropping detection
        self.detect_eva()
        #  Count the vote
        for i in range(self.v_num):
            y.append([])
            for j in range(self.param):
                y[i].append(np.mod(self.ballot_vector[i][j] +
                                   self.r[i][self.eff[int(self.p_num/2 - self.param) + j]], 2))
        for col in zip(*y):  # 二维列表取列
            ys.append(np.mod(sum(col), 2))
        for j in range(self.param):
            if self.mea_1[int(self.p_num/2 - self.param) + j].all() \
                    == self.mea_2[int(self.p_num/2 - self.param) + j].all():
                w.append(ys[j])
            else:
                w.append(np.mod(ys[j] + 1, 2))
            if w[j] == 1:
                print("Event rejected")
                return w[j]
        if self.flag == 1 and sum(w) == 0:
            print("Event rejected,even number")
            return 1
        else:
            print("Event pass")
            return 0

    def detect_eva(self):
        # Check that the number of valid events is appropriate
        num = 0
        self.eff = []  # Column number, the label corresponding to the single photon satisfying the valid condition
        q = int(self.p_num/2 - self.param)
        r_col_sum = []  # Record the sum of each column of r
        for col in zip(*self.s):
            if np.mod(sum(col), 2) == 0:
                self.eff.append(num)
            num += 1
        if len(self.eff) < self.param + q:
            print("Can't get the result")
            return
        for col in zip(*self.r):
            r_col_sum.append(np.mod(sum(col), 2))
        for i in range(q):
            if r_col_sum[self.eff[i]] == 0 and self.mea_1[self.eff[i]].all() != self.mea_2[self.eff[i]].all():
                print("Eavesdropper detected")
                break
            elif r_col_sum[self.eff[i]] == 1 and self.mea_1[self.eff[i]].all() == self.mea_2[self.eff[i]].all():
                print("Eavesdropper detected")
                break









