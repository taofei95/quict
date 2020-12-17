#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# @Time    :   2020/12/16 20:51:46
# @Author  :   Jia Zhang 
# @Contact :   jialrs.z@gmail.com
# @File    :   experience_pool.py

import numpy as np
class ExperiencePool(object):
    def __init__(self, max_capacity = 20000):
        self.max_capacity = max_capacity
        self._father_list = []
        self._children_list = []
        self._rewards_list = []
        self._sim_val = []

    def push(self, father, children, rewards, sim_val = 0):
        while len(self._father_list) >= self.max_capacity:
            self.pop() # Maybe to pop random one instead of the first one
        self._father_list.append(father)
        self._children_list.append(children)
        self._rewards_list.append(rewards)
        self._sim_val.append(sim_val)

    def update_sim_val(self, index, sim_val):
        self._sim_val[index] = sim_val

    def pop(self, index = 0):
        self._father_list.pop(index)

    def get_batch(self, batch_size = 32):
        indices = np.random.choice(range(len(self._father_list), batch_size))
        return np.choose(indices, self._father_list), \
               np.choose(indices, self._children_list), \
               np.choose(indices, self._rewards_list), \
               np.choose(indices, self._sim_val)