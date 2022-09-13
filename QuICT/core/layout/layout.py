#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/12/27 5:03 下午
# @Author  : Han Yu
# @File    : topology.py

import warnings


class LayoutEdge:
    """ Implement a physical connection between physical qubits

    Attributes:
        u(int): the node u of edge
        v(int): the node v of edge
        error_rate(float): the error_rate between u and v, default 1

    """

    @property
    def u(self) -> int:
        return self._u

    @u.setter
    def u(self, u):
        warnings.warn("in general, the LayoutEdge shouldn't be wrote.")
        self._u = u

    @property
    def v(self) -> int:
        return self._v

    @v.setter
    def v(self, v):
        warnings.warn("in general, the LayoutEdge shouldn't be wrote.")
        self._v = v

    @property
    def error_rate(self) -> float:
        return self._error_rate

    @error_rate.setter
    def error_rate(self, error_rate):
        self._error_rate = error_rate

    def __init__(self, u, v, error_rate):
        self._u = u
        self._v = v
        self._error_rate = error_rate


class Layout:
    """ Implement a topology in a physical device

    Attributes:
        name(string): the name of the topology
        edge_list(list<LayoutEdge>): the edge in layout
        qubit_number(int): the number of qubits
    """

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def edge_list(self) -> list:
        return self._edge_list

    @property
    def qubit_number(self):
        return self._qubit_number

    def __init__(self, n, name="unkown"):
        self._qubit_number = n
        self._name = name
        self._edge_list = []

    def __str__(self):
        return f"{self._name}\n{self._qubit_number}\n{self._edge_list}"

    def out_edges(self, begin_point) -> list:
        """ edges begin from begin_point

        Args:
            begin_point(int): the index of begin node

        Return:
            list<LayoutEdge>: edges begin from begin_point
        """
        prior_list = []
        for edge in self.edge_list:
            if edge.u == begin_point:
                prior_list.append(edge)
        return prior_list

    def add_edge(self, u, v, error_rate=1.0, directional=False):
        """ add an edge in the layout

        Args:
            u(int): the edge endpoint u
            v(int): the edge endpoint v
            error_rate(float): the error_rate, default 1
            two_way(bool): whether the edge is directional
        """
        if u == v:
            raise Exception("two endpoint shouldn't be the same")
        edge = LayoutEdge(u, v, error_rate)
        self._inner_add_edge(edge)
        if not directional:
            edge = LayoutEdge(v, u, error_rate)
            self._inner_add_edge(edge)

    def check_edge(self, u, v):
        """ check whether layout contain u->v

        Args:
            u(int): the edge endpoint u
            v(int): the edge endpoint v
        Return:
            bool: whether layout contain u->v
        """
        for edge in self.edge_list:
            if edge.u == u and edge.v == v:
                return True
        return False

    def write_file(self, path='./'):
        """ write file

        Args:
            path(str): the path file in, default './'
        """
        with open(f"{path}{self.name}.layout", "w") as f:
            f.write(f"{self.name}\n")
            f.write(f"{self.qubit_number}\n")
            for edge in self.edge_list:
                if edge.error_rate == 1.0:
                    f.write(f"{edge.u} {edge.v}\n")
                else:
                    f.write(f"{edge.u} {edge.v} {edge.error_rate}\n")

    @staticmethod
    def load_file(file):
        """

        Args:
            file(str): the path of file
        Return:
            Layout: the layout generatored
        """
        with open(file) as f:
            name = f.readline().replace("\n", "").replace("\r", "").replace(" ", "")
            n = int(f.readline())
            layout = Layout(n, name)
            layout.name = name
            lines = f.readlines()
            for item in lines:
                numbers = item.split()
                if len(numbers) == 3:
                    u = int(numbers[0])
                    v = int(numbers[1])
                    error_rate = float(numbers[2])
                    layout.add_edge(u, v, error_rate)
                elif len(numbers) == 2:
                    u = int(numbers[0])
                    v = int(numbers[1])
                    layout.add_edge(u, v)
        return layout

    def _inner_add_edge(self, edge: LayoutEdge):
        """ add/replace edge in layout

        Args:
            edge(LayoutEdge): the edge to be added
        """
        for check in self.edge_list:
            if check.u == edge.u and check.v == edge.v:
                check.error_rate = edge.error_rate
                return
        self.edge_list.append(edge)
