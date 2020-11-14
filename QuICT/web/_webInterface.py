#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/6/8 9:09 下午
# @Author  : Han Yu
# @File    : _webInterface.py

from QuICT.algorithm import *
from QuICT.models import *

class WebInterface(object):
    @staticmethod
    def load_object(ast):
        size = ast["circuit"]["size"]
        name = ast["circuit"]["name"]
        string_gate = ast["gates"]
        circuit = Circuit(size)
        circuit.name = name

        gates = []

        for sgate in string_gate:
            name = sgate["name"]
            bits = sgate["bits"]
            if name == "H":
                GateBuilder.setGateType(GateType.H)
            elif name == "S":
                GateBuilder.setGateType(GateType.S)
            elif name == "S_dagger":
                GateBuilder.setGateType(GateType.S_dagger)
            elif name == "X":
                GateBuilder.setGateType(GateType.X)
            elif name == "Y":
                GateBuilder.setGateType(GateType.Y)
            elif name == "Z":
                GateBuilder.setGateType(GateType.Z)
            elif name == "ID":
                GateBuilder.setGateType(GateType.ID)
            elif name == "U1":
                GateBuilder.setGateType(GateType.U1)
            elif name == "U2":
                GateBuilder.setGateType(GateType.U2)
            elif name == "U3":
                GateBuilder.setGateType(GateType.U3)
            elif name == "Rx":
                GateBuilder.setGateType(GateType.Rx)
            elif name == "Ry":
                GateBuilder.setGateType(GateType.Ry)
            elif name == "Rz":
                GateBuilder.setGateType(GateType.Rz)
            elif name == "T":
                GateBuilder.setGateType(GateType.T)
            elif name == "T_dagger":
                GateBuilder.setGateType(GateType.T_dagger)
            elif name == "CZ":
                GateBuilder.setGateType(GateType.CZ)
            elif name == "CX":
                GateBuilder.setGateType(GateType.CX)
            elif name == "CH":
                GateBuilder.setGateType(GateType.CH)
            elif name == "CRz":
                GateBuilder.setGateType(GateType.CRz)
            elif name == "CCX":
                GateBuilder.setGateType(GateType.CCX)
            elif name == "Measure":
                GateBuilder.setGateType(GateType.Measure)
            elif name == "Swap":
                GateBuilder.setGateType(GateType.Swap)
            cargs = GateBuilder.getCargsNumber()
            pargs = GateBuilder.getParamsNumber()
            if cargs > 0:
                GateBuilder.setCargs(bits[:cargs])
            GateBuilder.setTargs(bits[cargs:])
            params = []
            if pargs > 0:
                parg = sgate["args"]
                for p in parg:
                   params.append(p * np.pi)
                GateBuilder.setPargs(params)
            gates.append(GateBuilder.getGate())
        circuit.set_flush_gates(gates)
        return circuit

    @staticmethod
    def output_object(circuit : Circuit):
        ast = {"circuit" : {}}
        ast["circuit"]["size"] = circuit.circuit_size()
        ast["circuit"]["name"] = circuit.name
        gates = []
        for gate in circuit.gates:
            g = {"name" : str(gate)[:-1]}
            bits = []
            for c in gate.cargs:
                bits.append(c)
            for q in gate.targs:
                bits.append(q)
            args = []
            for p in gate.pargs:
                args.append((p / np.pi))
            g["bits"] = bits
            g["args"] = args
            gates.append(g)
        ast["gates"] = gates
        return ast

    @staticmethod
    def simplification(ast, method):
        circuit = WebInterface.load_object(ast)
        if method == "CNOT_Rz":
            circuit = CNOT_RZ.run(circuit)
        else:
            raise Exception("该算法尚无法用图形化界面调用")
        return WebInterface.output_object(circuit)
