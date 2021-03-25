#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2021/3/24 9:36 下午
# @Author  : Han Yu
# @File    : gate_set

from .gate import *


class GateSet(list):
    """ Implement a list of gate

    Attributes:
        gates:(list<BasicGate>): GateSet itself

    """

    @property
    def gates(self):
        return self

    def __enter__(self):
        GATE_SET_LIST.append(self)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global GATE_SET_LIST
        GATE_SET_LIST.remove(self)

    def __init__(self, gates = None):
        """ initial a GateSet with gate(s)

        Args:
            qubits: the qubits which make up the qureg, it can have below form,
                1) Circuit
                2) BasicGate
                3) GateSet
                4) tuple/list<BasicGate>
        """
        super().__init__()
        if gates is None:
            return
        if isinstance(gates, BasicGate):
            self.append(gates)
        else:
            if isinstance(gates, Circuit):
                gates = gates.gates
            gates = list(gates)
            for gate in gates:
                self.append(gate)

    # Attributes of the circuit
    def circuit_width(self):
        """ the number of qubits applied by gates

        Returns:
            int: the number of qubits applied by gates
        """
        qubits = set()
        for gate in self:
            qubits.add(gate.affectArgs)
        return len(qubits)

    def circuit_size(self):
        """ the size of the gates

        Returns:
            int: the number of gates in gates
        """
        return len(self)

    def circuit_count_2qubit(self):
        """ the number of the two qubit gates in the set

        Returns:
            int: the number of the two qubit gates in the set
        """
        count = 0
        for gate in self:
            if gate.controls + gate.targets == 2:
                count += 1
        return count

    def circuit_count_1qubit(self):
        """ the number of the one qubit gates in the set

        Returns:
            int: the number of the one qubit gates in the set
        """
        count = 0
        for gate in self.gates:
            if gate.controls + gate.targets == 1:
                count += 1
        return count

    def circuit_count_gateType(self, gateType):
        """ the number of the gates which are some type in the set

        Args:
            gateType(GateType): the type of gates to be count

        Returns:
            int: the number of the gates which are some type in the circuit
        """
        count = 0
        for gate in self:
            if gate.type == gateType:
                count += 1
        return count

    def circuit_depth(self, gateTypes = None):
        """ the depth of the circuit for some gate.

        Args:
            gateTypes(list<GateType>):
                the types to be count into depth calculate
                if count all type of gates, leave it being None.

        Returns:
            int: the depth of the circuit
        """
        layers = []
        for gate in self:
            if gateTypes is None or gate.type in gateTypes:
                now = set(gate.cargs) | set(gate.targs)
                for i in range(len(layers) - 1, -2, -1):
                    if i == -1 or len(now & layers[i]) > 0:
                        if i + 1 == len(layers):
                            layers.append(set())
                        layers[i + 1] |= now
        return len(layers)

    def qasm(self):
        """ get OpenQASM 2.0 describe for the circuit

        Returns:
            str: OpenQASM 2.0 describe or "error" when there are some gates cannot be resolved
        """
        qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        cbits = 0
        for gate in self.gates:
            if gate.qasm_name == "measure":
                cbits += 1
        qasm_string += f"qreg q[{self.circuit_width()}];\n"
        if cbits != 0:
            qasm_string += f"creg c[{cbits}];\n"
        cbits = 0
        for gate in self.gates:
            if gate.qasm_name == "measure":
                qasm_string += f"measure q[{gate.targ}] -> c[{cbits}];\n"
                cbits += 1
            else:
                qasm = gate.qasm()
                if qasm == "error":
                    print("the circuit cannot be transformed to a valid describe in OpenQASM 2.0")
                    gate.print_info()
                    return "error"
                qasm_string += gate.qasm()
        return qasm_string

    def __or__(self, targets):
        """deal the operator '|'

        Use the syntax "gateSet | circuit" or "gateSet | qureg" or "gateSet | qubit"
        to add the gate of gateSet into the circuit/qureg/qubit

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) Circuit
                2) qureg
                3) tuple<qubit, qureg>
                4) list<qubit, qureg>
        Raise:
            TypeException: the type of other is wrong
        """

        if isinstance(targets, tuple):
            targets = list(targets)
        if isinstance(targets, list):
            qureg = Qureg()
            for item in targets:
                if isinstance(item, Qubit):
                    qureg.append(item)
                elif isinstance(item, Qureg):
                    qureg.extend(item)
                else:
                    raise TypeException("qubit or tuple<qubit, qureg> or qureg or list<qubit, qureg> or circuit", targets)
        elif isinstance(targets, Qureg):
            qureg = targets
        elif isinstance(targets, Circuit):
            qureg = Qureg(targets.qubits)
        else:
            raise TypeException("qubit or tuple<qubit> or qureg or circuit", targets)

        self.targets = len(qureg)

        for gate in gates:
            qubits = []
            for control in gate.cargs:
                qubits.append(qureg[control])
            for target in gate.targs:
                qubits.append(qureg[target])
            qureg.circuit.append(gate, qubits)

        for gate in self.gates:
            gate | targets

    def __xor__(self, targets):
        """deal the operator '^'

        Use the syntax "gateSet ^ circuit" or "gateSet ^ qureg" or "gateSet ^ qubit"
        to add the gate of gateSet's inverse into the circuit/qureg/qubit

        Note that the order of qubits is that control bits first
        and target bits followed.

        Args:
            targets: the targets the gate acts on, it can have following form,
                1) Circuit
                2) qureg
                3) tuple<qubit, qureg>
                4) list<qubit, qureg>
        Raise:
            TypeException: the type of other is wrong
        """


        for gate in self.inverse():
            gate | targets

    def __getitem__(self, item):
        """ to fit the slice operator, overloaded this function.

        get a smaller qureg/qubit from this qureg

        Args:
            item(int/slice): slice passed in.
        Return:
            Qubit/Qureg: the result or slice
        """
        if isinstance(item, int):
            return super().__getitem__(item)
        elif isinstance(item, slice):
            gate_list = super().__getitem__(item)
            return GateSet(gate_list)

    def __add__(self, other):
        """ to fit the add operator, overloaded this function.

        get a smaller qureg/qubit from this qureg

        Args:
            other(Qureg): qureg to be added.
        Return:
            Qureg: the result or slice
        """
        if not isinstance(other, Qureg):
            raise Exception("type error!")
        gate_list = super().__add__(other)
        return GateSet(gate_list)

    # display information of the circuit
    def print_infomation(self):
        print("-------------------")
        print(f"number of bits:{self.circuit_width()}")
        for gate in self:
            gate.print_info()
        print("-------------------")

    def random_append(self, rand_size = 10, typeList = None):
        """ add some random gate to the circuit
        Args:
            rand_size(int): the number of the gate added to the circuit
            typeList(list<GateType>): the type of gate, default contains CX、ID、Rz、CY、CRz、CH
        """
        inner_random_append(self, rand_size, typeList)

    def inverse(self):
        """ the inverse of GateSet

        Returns:
            GateSet: the inverse of the gateSet
        """
        inverse = GateSet()
        circuit_size = len(self)
        for index in range(circuit_size - 1, -1, -1):
            inverse.append(self[index])
        return inverse
