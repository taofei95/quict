#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/3/22 10:31 上午
# @Author  : Han Yu
# @File    : _qasmInterface.py

from ._basicInterface import BasicInterface
from QuICT.core import *
from QuICT.lib import Qasm
from QuICT.core.exception import QasmInputException
from collections import OrderedDict
from configparser import ConfigParser
import os

class qasm_qreg(object):
    """ the target bit in the qasm grammer tree

    """
    def __init__(self, index_list, name):
        self.index = index_list
        self.name = name

class qasm_creg(object):
    """ the control bit in the qasm grammer tree

    """
    def __init__(self, index_list, name):
        self.index = index_list
        self.name = name

class OPENQASMInterface(BasicInterface):
    # qasm mapping to QuICT
    standard_extension = {"u1": GateType.U1,
                          "u2": GateType.U2,
                          "u3": GateType.U3,
                          "U": GateType.U3,
                          "x": GateType.X,
                          "y": GateType.Y,
                          "z": GateType.Z,
                          "t": GateType.T,
                          "tdg": GateType.T_dagger,
                          "s": GateType.S,
                          "sdg": GateType.S_dagger,
                          "swap": GateType.Swap,
                          "rx": GateType.Rx,
                          "ry": GateType.Ry,
                          "rz": GateType.Rz,
                          "id": GateType.ID,
                          "h": GateType.H,
                          "cx": GateType.CX,
                          "ccx" : GateType.CCX,
                          "cy": GateType.CY,
                          "cz": GateType.CZ,
                          "ch": GateType.CH,
                          "crz": GateType.CRz}

    extern_extension = {
                          "rzz" : ExtensionGateType.RZZ,
                          "cu1" : ExtensionGateType.CU1,
                          "cu3" : ExtensionGateType.CU3,
                          "cswap" : ExtensionGateType.Fredkin
    }

    token = None

    DEFAULT_QUICT_PATH = os.path.join(os.path.expanduser("~"),
                                      '.QuICT')

    DEFAULT_QUICT_FILE = os.path.join(os.path.expanduser("~"),
                                         '.QuICT', 'accounts.ini')

    @staticmethod
    def load_circuit(circuit : Circuit):
        instance = OPENQASMInterface()
        instance.circuit = circuit
        instance.analyse_code_from_circuit()
        return instance

    @staticmethod
    def load_file(filename : str):
        instance = OPENQASMInterface()
        instance.ast = Qasm(filename).parse()
        instance.analyse_circuit_from_ast(instance.ast)
        return instance

    def __init__(self):
        super().__init__()
        self.circuit = None
        self.ast = None
        self.qasm = None

        self.qbits = 0
        self.cbits = 0
        self.valid_circuit = False
        self.valid_qasm = False
        self.gates = OrderedDict()
        self.node_gates = []
        self.qregs = {}
        self.cregs = {}
        self.arg_stack = [{}]
        self.bit_stack = [{}]
        self.circuit_gates = []
        self.version = None

    def analyse_circuit_from_ast(self, node):
        self.valid_circuit = True
        self.analyse_node(node)
        if self.valid_circuit:
            self.circuit = Circuit(self.qbits)
            self.circuit.set_exec_gates(self.circuit_gates)

    def analyse_code_from_circuit(self):
        self.valid_qasm = True
        self.qasm = self.circuit.qasm()
        if self.qasm == "error":
            self.qasm = None
            self.valid_qasm = False

    def output_qasm(self, filename = None):
        if not self.valid_qasm or self.qasm is None:
            if self.circuit is None:
                return
            self.qasm = self.circuit.qasm()
            if self.qasm != 'error':
                self.valid_qasm = True
            else:
                return False
        if filename is None:
            print(self.qasm)
        else:
            with open(filename, 'w+') as file:
                file.write(self.qasm)

    def enable_path(self):
        if not os.path.exists(self.DEFAULT_QUICT_PATH):
            os.mkdir(self.DEFAULT_QUICT_PATH)

    def load_token(self):
        self.enable_path()
        config_parser = ConfigParser()
        config_parser.read(self.DEFAULT_QUICT_FILE)
        if 'account' in config_parser:
            dic = dict(config_parser.items('account'))
            if 'token' in dic:
                self.token = dic['token']

    def save_token(self, token):
        self.token = token
        self.enable_path()
        config_parser = ConfigParser()
        config_parser['account'] = {
            'token' : token,
        }
        with open(self.DEFAULT_QUICT_FILE, 'w') as f:
            config_parser.write(f)

    def output_qiskit(self, filename, generator_qasm = False, shots = 1024):
        if not self.valid_qasm or self.qasm is None:
            if self.circuit is None:
                return
            self.qasm = self.circuit.qasm()
            if self.qasm != 'error':
                self.valid_qasm = True
            else:
                return False

        if self.token is None:
            self.load_token()
            if self.token is None:
                token = input("please input token > ")
                self.save_token(token)

        if generator_qasm:
            with open(filename + '.qasm', 'w+') as file:
                file.write(self.qasm)
            code = """
from qiskit import QuantumCircuit
from qiskit import IBMQ, execute
from qiskit.providers.ibmq import least_busy
IBMQ.save_account('{}', overwrite=True)
circ = QuantumCircuit.from_qasm_file("{}.qasm")
provider = IBMQ.load_account()
least_busy_device = least_busy(
provider.backends(simulator=False,
    filters=lambda x: x.configuration().n_qubits > 4))
job = execute(circ, least_busy_device, shots={})
result = job.result()
print(result.get_counts(circ))
""".format(self.token, filename, shots)
            with open(filename + '.py', 'w+') as file:
                file.write(code)
        else:
            code = """
from qiskit import QuantumCircuit
from qiskit import IBMQ, execute
from qiskit.providers.ibmq import least_busy
IBMQ.save_account('{}')
circ = QuantumCircuit.from_qasm_str({})
provider = IBMQ.load_account()
least_busy_device = least_busy(
provider.backends(simulator=False,
    filters=lambda x: x.configuration().n_qubits > 4))
job = execute(circ, least_busy_device, shots={})
result = job.result()
print(result.get_counts(circ))
""".format(self.token, '"""' + self.qasm + '"""', shots)
            with open(filename + '.py', 'w+') as file:
                file.write(code)

    def analyse_node(self, node):
        if not self.valid_circuit:
            return
        if node.type == "program":
            for child in node.children:
                self.analyse_node(child)

        elif node.type == "qreg":
            self.qregs[node.name] = qasm_qreg([i for i in range(self.qbits, self.qbits + node.index)], node.name)
            self.qbits += node.index

        elif node.type == "creg":
            self.cregs[node.name] = qasm_creg([i for i in range(self.cbits, self.cbits + node.index)], node.name)
            self.cbits += node.index

        elif node.type == "id_list":
            return [self.get_analyse_id(child) for child in node.children]

        elif node.type == "primary_list":
            return [self.get_analyse_id(child) for child in node.children]

        elif node.type == "gate":
            self.analyse_gate(node)

        elif node.type == "custom_unitary":
            self.analyse_custom(node)

        elif node.type == "universal_unitary":
            QasmInputException("universal_unitary is deprecated", node.line, node.file)

        elif node.type == "cnot":
            self.analyse_cnot(node)

        elif node.type == "expression_list":
            return node.children

        elif node.type == "measure":
            self.analyse_measure(node)

        elif node.type == "format":
            self.version = node.version()

        elif node.type == "barrier":
            pass

        elif node.type == "reset":
            self.analyse_reset(node)

        elif node.type == "if":
            self.analyse_if(node)

        elif node.type == "opaque":
            self.analyse_opaque(node)

        else:
            QasmInputException("QASM grammer error", node.line, node.file)

    def analyse_gate(self, node):
        self.gates[node.name] = {}
        de_gate = self.gates[node.name]
        de_gate["n_args"] = node.n_args()
        de_gate["n_bits"] = node.n_bits()
        if node.n_args() > 0:
            de_gate["args"] = [element.name for element in node.arguments.children]
        else:
            de_gate["args"] = []
        de_gate["bits"] = [child.name for child in node.bitlist.children]
        if node.name in self.standard_extension or node.name in self.extern_extension:
            return
        de_gate["body"] = node.body

    def analyse_custom(self, node):
        name = node.name
        if node.arguments is not None:
            args = self.analyse_node(node.arguments)
        else:
            args = []
        bits = [
            self.get_analyse_id(node_element)
            for node_element in node.bitlist.children
        ]
        if name in self.gates:
            gargs = self.gates[name]["args"]
            gbits = self.gates[name]["bits"]

            maxidx = max(map(len, bits))
            for idx in range(maxidx):
                self.arg_stack.append({gargs[j]: args[j]
                                       for j in range(len(gargs))})
                element = [idx * x for x in
                           [len(bits[j]) > 1 for j in range(len(bits))]]
                self.bit_stack.append({gbits[j]: bits[j][element[j]]
                                       for j in range(len(gbits))})
                self.analyse_name(name,
                                   gargs,
                                   gbits)
                self.arg_stack.pop()
                self.bit_stack.pop()
        else:
            raise QasmInputException("undefined gate:", node.line, node.file)

    def analyse_cnot(self, node):
        id0 = self.get_analyse_id(node.children[0])
        id1 = self.get_analyse_id(node.children[1])
        if not (len(id0) == len(id1) or len(id0) == 1 or len(id1) == 1):
            raise QasmInputException("the number of bits unmatched:", node.line, node.file)

        maxidx = max([len(id0), len(id1)])
        GateBuilder.setGateType(GateType.CX)
        for idx in range(maxidx):
            if len(id0) > 1 and len(id1) > 1:
                GateBuilder.setCargs(id0[idx])
                GateBuilder.setTargs(id1[idx])
                self.circuit_gates.append(GateBuilder.getGate())
            elif len(id0) > 1:
                GateBuilder.setCargs(id0[idx])
                GateBuilder.setTargs(id1[0])
                self.circuit_gates.append(GateBuilder.getGate())
            else:
                GateBuilder.setCargs(id0[0])
                GateBuilder.setTargs(id1[idx])
                self.circuit_gates.append(GateBuilder.getGate())

    def analyse_measure(self, node):
        id0 = self.get_analyse_id(node.children[0])
        id1 = self.get_analyse_id(node.children[1])
        if len(id0) != len(id1):
            raise QasmInputException("the number of bits of registers unmatched:", node.line, node.file)

        GateBuilder.setGateType(GateType.Measure)
        for idx, _ in zip(id0, id1):
            GateBuilder.setTargs(idx)
            self.circuit_gates.append(GateBuilder.getGate())

    def analyse_reset(self, node):
        id0 = self.get_analyse_id(node.children[0])
        GateBuilder.setGateType(GateType.Reset)
        for i, _ in enumerate(id0):
            GateBuilder.setTargs(id0[i])
            self.circuit_gates.append(GateBuilder.getGate())

    def analyse_if(self, node):
        print("if op is not supported:{}" ,node.type)
        self.valid_circuit = False

    def analyse_opaque(self, node):
        pass

    def analyse_name(self, name, gargs, gbits):
        if name in self.standard_extension:
            pargs = [self.arg_stack[-1][s].sym(self.arg_stack[:-1]) for s in gargs]
            targs = [self.bit_stack[-1][s] for s in gbits]
            type = self.standard_extension[name]
            GateBuilder.setGateType(type)
            GateBuilder.setPargs(pargs)
            GateBuilder.setArgs(targs)
            self.circuit_gates.append(GateBuilder.getGate())
        elif name in self.extern_extension:
            pargs = [self.arg_stack[-1][s].sym(self.arg_stack[:-1]) for s in gargs]
            targs = [self.bit_stack[-1][s] for s in gbits]
            ExtensionGateBuilder.setGateType(self.extern_extension[name])
            ExtensionGateBuilder.setPargs(pargs)
            ExtensionGateBuilder.setTargs(targs)
            self.circuit_gates.extend(ExtensionGateBuilder.getGate())
        else:
            body = self.gates[name]['body']
            for child in body.children:
                if child.arguments is not None:
                    args = self.analyse_node(child.arguments)
                else:
                    args = []
                bits = [
                    self.get_analyse_id(node_element)
                    for node_element in child.bitlist.children
                ]
                gargs = self.gates[child.name]["args"]
                gbits = self.gates[child.name]["bits"]

                maxidx = max(map(len, bits))
                for idx in range(maxidx):
                    self.arg_stack.append({gargs[j]: args[j]
                                           for j in range(len(gargs))})
                    element = [idx * x for x in
                               [len(bits[j]) > 1 for j in range(len(bits))]]
                    self.bit_stack.append({gbits[j]: bits[j][element[j]]
                                           for j in range(len(gbits))})
                    self.analyse_name(child.name,
                                      gargs,
                                      gbits)
                    self.arg_stack.pop()
                    self.bit_stack.pop()

    def get_analyse_id(self, node):
        if node.name in self.qregs:
            reg = self.qregs[node.name]
        elif node.name in self.cregs:
            reg = self.cregs[node.name]
        elif self.bit_stack[-1] and node.name in self.bit_stack[-1]:
            reg = self.bit_stack[-1][node.name]
        else:
            raise QasmInputException("expected qreg or creg name:", node.line, node.file)

        if node.type == "indexed_id":
            return [reg.index[node.index]]
        elif node.type == "id":
            if not self.bit_stack[-1]:
                return [bit for bit in reg.index]
            else:
                if node.name in self.bit_stack[-1]:
                    return [self.bit_stack[-1][node.name]]
                raise QasmInputException("expected qreg or creg name:", node.line, node.file)
        return None
