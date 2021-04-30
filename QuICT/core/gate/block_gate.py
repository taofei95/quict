#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/4/27 1:14 下午
# @Author  : Han Yu
# @File    : block_gate

from .gate import *

class BlockGate(object):

    def __init__(self):
        self.__width = 0
        self.__size = 0
        self.__affectBits = set()
        self.__gates = []

    @property
    def width(self):
        return self.__width

    @property
    def size(self):
        return self.__size

    @property
    def affectBits(self):
        return self.__affectBits

    @property
    def gates(self):
        return self.__gates

    def spatialGateBits(self, target: BasicGate) -> int:
        count = self.width
        for bit in target.affectArgs:
            if bit not in self.affectBits:
                count += 1
        return count

    def spatialBlockBits(self, target) -> int:
        count = self.width
        for bit in target.affectBits:
            if bit not in self.affectBits:
                count += 1
        return count

    def addGate(self, target: BasicGate, bits = None):
        if bits is None:
            bits = self.spatialGateBits(target)
        self.__width = bits
        self.__size += 1
        self.__gates.append(target)

    def extendBlock(self, target, bits = None):
        if bits is None:
            bits = self.spatialBlockBits(target)
        self.__width = bits
        self.__size += 1
        self.__gates.extend(target.gates)

