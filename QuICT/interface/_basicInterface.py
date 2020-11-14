#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/3/15 1:38 下午
# @Author  : Han Yu
# @File    : _basicInterface.py

from QuICT.models import *

class BasicInterface(object):

    # 对应电路
    @property
    def circuit(self):
        return self.__circuit

    @circuit.setter
    def circuit(self, circuit):
        self.__circuit = circuit

    # 对应文本内容
    @property
    def text_content(self):
        return self.__text_content

    @text_content.setter
    def text_content(self, text_content):
        self.__text_content = text_content

    def __init__(self):
        self.circuit = None
        self.text_content = None

    @staticmethod
    def load_circuit(circuit : Circuit):
        """
        :param circuit: 待加载的电路
        :return: BasicInterface的一个实例
        """
        instance = BasicInterface()
        instance.circuit = circuit
        return instance

    @staticmethod
    def load_file(filename : str):
        """
        :param filename: 文件名称
        :return: BasicInterface的一个实例
        """

        instance = BasicInterface()
        with open(filename) as file:
            instance.text_content = file.read()
        return instance
