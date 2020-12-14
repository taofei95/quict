#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/6/8 9:09
# @Author  : Han Yu
# @File    : _webInterface.py

import numpy as np

from QuICT.QCDA.optimization import *
from QuICT.core import *

class WebInterface(object):
    """ The model servers for Web Interface for QuICT

    """
    @staticmethod
    def load_object(ast):
        """ load data from js and generator a circuit

        Args:
            ast(dict): data from js, describe a circuit
        Returns:
            Circuit: the circuit generated by ast

        """
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
                GateBuilder.setGateType(GATE_ID["H"])
            elif name == "S":
                GateBuilder.setGateType(GATE_ID["S"])
            elif name == "S_dagger":
                GateBuilder.setGateType(GATE_ID["S_dagger"])
            elif name == "X":
                GateBuilder.setGateType(GATE_ID["X"])
            elif name == "Y":
                GateBuilder.setGateType(GATE_ID["Y"])
            elif name == "Z":
                GateBuilder.setGateType(GATE_ID["Z"])
            elif name == "ID":
                GateBuilder.setGateType(GATE_ID["ID"])
            elif name == "U1":
                GateBuilder.setGateType(GATE_ID["U1"])
            elif name == "U2":
                GateBuilder.setGateType(GATE_ID["U2"])
            elif name == "U3":
                GateBuilder.setGateType(GATE_ID["U3"])
            elif name == "Rx":
                GateBuilder.setGateType(GATE_ID["Rx"])
            elif name == "Ry":
                GateBuilder.setGateType(GATE_ID["Ry"])
            elif name == "Rz":
                GateBuilder.setGateType(GATE_ID["Rz"])
            elif name == "T":
                GateBuilder.setGateType(GATE_ID["T"])
            elif name == "T_dagger":
                GateBuilder.setGateType(GATE_ID["T_dagger"])
            elif name == "CZ":
                GateBuilder.setGateType(GATE_ID["CZ"])
            elif name == "CX":
                GateBuilder.setGateType(GATE_ID["CX"])
            elif name == "CH":
                GateBuilder.setGateType(GATE_ID["CH"])
            elif name == "CRz":
                GateBuilder.setGateType(GATE_ID["CRz"])
            elif name == "CCX":
                GateBuilder.setGateType(GATE_ID["CCX"])
            elif name == "Measure":
                GateBuilder.setGateType(GATE_ID["Measure"])
            elif name == "Swap":
                GateBuilder.setGateType(GATE_ID["Swap"])
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
        circuit.set_exec_gates(gates)
        return circuit

    @staticmethod
    def output_object(circuit : Circuit):
        """ transform circuit form to js data dict
        Args:

        Returns:

        """
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
        """ call the simplification algorithm in QuICT

        Args:
            ast(dict): data from js, describe a circuit
            method(str): the name of optimization
        Returns:
            dict: data to js, describe a circuit after optimization
        """
        circuit = WebInterface.load_object(ast)
        if method == "CNOT_Rz":
            circuit = topological_cnot_rz.run(circuit)
        else:
            raise Exception("it is not supported now")
        return WebInterface.output_object(circuit)
