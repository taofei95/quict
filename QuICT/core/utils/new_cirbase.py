#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/15 10:31
# @Author  : Han Yu, Li Kaiqi
# @File    : _circuit_computing.py
from collections import defaultdict
from enum import Enum
import numpy as np

from .gate_type import GateType
from .variable import Variable


class CircuitGates:
    @property
    def gate_counts(self) -> tuple:
        return (self._singleq_gates_count, self._biq_gates_count, self._gate_type_count)

    @property
    def gates(self) -> list:
        return [gate.copy() & targs for gate, targs, _ in self._gates]

    @property
    def size(self) -> int:
        return self._size

    @property
    def length(self) -> int:
        return len(self._gates)

    @property
    def singleq_gates_count(self) -> int:
        return self._singleq_gates_count

    @property
    def biq_gates_count(self) -> int:
        return self._biq_gates_count

    def gates_count_by_type(self, gate_type: GateType) -> int:
        return self._gate_type_count[gate_type]

    @property
    def training_gates_count(self) -> int:
        return self._tgates_count

    def __init__(self, qubits: int = None):
        self._gates = []
        self._limit_qubits = qubits
        self._pointer = None

        self._initial_property()

    def _initial_property(self):
        # Gates's Property
        self._size = 0
        self._biq_gates_count = 0
        self._singleq_gates_count = 0
        self._tgates_count = 0
        self._gate_type_count = defaultdict(int)

    def __call__(self):
        return self._gates

    def _validate_qidxes(self, qidxes: list):
        for qidx in qidxes:
            assert qidx < self._limit_qubits, ValueError("Gates' qubit indexes exceed circuit limited.")

    def append(self, gate, qidxes: list, size: int):
        if self._limit_qubits is not None:
            self._validate_qidxes(qidxes)

        self._gates.append((gate, qidxes, size))
        self._size += size
        self._analysis_gate(gate)

    def pop(self, idx: int):
        gate, qidxes, size = self._gates.pop(idx)
        self._analysis_gate(gate, minus=-1)
        self._size -= size

        return (gate, qidxes, size)

    def _analysis_gate(self, gate, minus: int = 1):
        # Gates count update
        if gate.controls + gate.targets == 1:
            self._singleq_gates_count += minus
        elif gate.controls + gate.targets == 2:
            self._biq_gates_count += minus

        if gate.variables > 0:
            self._tgates_count += minus

        self._gate_type_count[gate.type] += minus

    def depth(self) -> list:
        """ Return the depth of the circuit.

        Returns:
            list: The Depth for each qubits
        """
        circuit_depth = defaultdict(int)
        for gate, qidxes, _ in self._gates:
            if type(gate).__name__ == "CompositeGate":
                gate_depth = gate._gates.depth
                for i in len(gate_depth):
                    circuit_depth[qidxes[i]] += gate_depth[i]
            else:
                pre_depth = [circuit_depth[q] for q in qidxes]
                new_depth = max(pre_depth) + 1
                for q in qidxes:
                    circuit_depth[q] = new_depth

        if self._limit_qubits is not None:
            qubits = list(range(self._limit_qubits))
        else:
            qubits = list(circuit_depth.keys())
            qubits.sort()

        return [circuit_depth[qubit] for qubit in qubits]

    def get_flatten_gates(self):
        flat_gates = []
        for gate, qidxes, size in self._gates:
            if type(gate).__name__ == "CompositeGate":
                temp_gate = gate.copy() & qidxes
                flat_gates += temp_gate.flatten_gates()
            else:
                flat_gates.append((gate.copy(), qidxes, size))

        return flat_gates

    def flatten(self):
        self._gates = self.get_flatten_gates()

    def get_decomposition_gates(self):
        decomp_gates = []
        for gate, qidxes, size in self._gates:
            if type(gate).__name__ == "CompositeGate":
                temp_gate = gate.copy() & qidxes
                decomp_gates += temp_gate.decomposition_gates()
            else:
                cgate = gate.build_gate(qidxes)
                if cgate is not None:
                    decomp_gates += cgate.fast_gates
                else:
                    decomp_gates.append((gate.copy(), qidxes, size))

        return decomp_gates

    def decomposition(self):
        self._gates = self.get_decomposition_gates()
        self._update_properties()

    def _update_properties(self):
        self._initial_property()
        for gate, _, size in self._gates:
            self._analysis_gate(gate)
            self._size += size


class CircuitBased(object):
    """ Based Class for Circuit and Composite Gate. """

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        self._name = name

    @property
    def precision(self) -> str:
        return self._precision

    ####################################################################
    ############         Circuit's Gates Function           ############
    ####################################################################
    @property
    def gates(self) -> list:
        """ Return the list of BasicGate/CompositeGate/Operator in the current circuit. \n
        *Warning*: this is slowly due to the copy of gates, you can use self.fast_gates to
        get list of tuple(gate, qubit_indexes, size) for further using.
        """
        return self._gates.gates

    @property
    def fast_gates(self) -> list:
        """ Return the list of tuple(gate, qubit_indexes, size) in the current circuit. """
        return self._gates

    def gate_decomposition(self, self_flatten: bool = True, decomposition: bool = True) -> list:
        """ Decomposition the CompositeGate or BasicGate which has build_gate function.

        Args:
            self_flatten (bool, optional): Whether change the gates in current Circuit. Defaults to True.
            decomposition (bool, optional): Whether call build_gate for BasicGates. Defaults to True.

        Returns:
            list: The list of (gate, qubit indexes, size)
        """
        return self._gates.get_decomposition_gates()

    def flatten_gates(self, decomposition: bool = False) -> list:
        """ Get the list of Quantum Gates with decompose the CompositeGate.

        Args:
            decomposition (bool, optional): Whether call build_gate for Quantum Gates. Defaults to False.

        Returns:
            List[BasicGate]: The list of BasicGate/Operator.
        """
        return self._gates.get_flatten_gates()

    def decomposition(self):
        self._gates.decomposition()

    def flatten(self):
        self._gates.flatten()

    def __init__(self, name: str, qubits: int = None):
        """
        Args:
            name (str): The name of current Quantum Circuit
        """
        self._name = name
        self._gates = CircuitGates(qubits)
        self._qubits = qubits
        self._pointer = None
        self._precision = "double"

    ####################################################################
    ############           Circuit's Properties             ############
    ####################################################################
    def size(self) -> int:
        """ the number of gates in the circuit/CompositeGate, the operators are not count.

        Returns:
            int: the number of gates in circuit
        """
        return self._gates.size

    def gate_length(self) -> int:
        return self._gates.length

    def width(self):
        """ The number of qubits in Circuit.

        Returns:
            int: the number of qubits in circuit
        """
        return len(self._qubits)

    def depth(self) -> int:
        """ the depth of the circuit.

        Returns:
            int: the depth
        """
        return np.max(self._gates.depth())

    def count_2qubit_gate(self) -> int:
        """ the number of the two qubit gates in the Circuit/CompositeGate

        Returns:
            int: the number of the two qubit gates
        """
        return self._gates.biq_gates_count

    def count_1qubit_gate(self) -> int:
        """ the number of the one qubit gates in the Circuit/CompositeGate

        Returns:
            int: the number of the one qubit gates
        """
        return self._gates.singleq_gates_count

    def count_gate_by_gatetype(self, gate_type: GateType) -> int:
        """ the number of the target Quantum Gate in the Circuit/CompositeGate

        Args:
            gateType(GateType): the type of gates to be count

        Returns:
            int: the number of the gates
        """
        return self._gates.gates_count_by_type(gate_type)

    def count_training_gate(self):
        """ the number of the trainable gates in the Circuit/CompositeGate

        Returns:
            int: the number of the trainable gates
        """
        return self._gates.training_gates_count

    def __str__(self):
        circuit_info = {
            "name": self.name,
            "width": self.width(),
            "size": self.size(),
            "depth": self.depth(),
            "1-qubit gates": self.count_1qubit_gate(),
            "2-qubit gates": self.count_2qubit_gate(),
            "training gates": self.count_training_gate(),
        }

        return str(circuit_info)

    def qasm(self, output_file: str = None):
        """ The qasm of current CompositeGate/Circuit. The Operator will be ignore.

        Args:
            output_file (str): The output qasm file's name or path

        Returns:
            str: The string of Circuit/CompositeGate's qasm.
        """
        # Header
        qreg = self.width()
        creg = min(self.count_gate_by_gatetype(GateType.measure), qreg)
        if creg == 0:
            creg = qreg

        qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        qasm_string += f"qreg q[{qreg}];\n"
        qasm_string += f"creg c[{creg}];\n"

        # Body [gates]
        cbits = 0
        for gate, targs, size in self.gate_decomposition(False, False):
            if size == 0:
                continue

            if gate.qasm_name == "measure":
                qasm_string += f"measure q[{targs[0]}] -> c[{cbits}];\n"
                cbits += 1
                cbits = cbits % creg
            else:
                qasm_string += gate.qasm(targs)

        if output_file is not None:
            with open(output_file, "w+") as of:
                of.write(qasm_string)

        return qasm_string

    def set_precision(self, precision: str):
        """ Set precision for Cicuit/CompositeGate

        Args:
            precision(str): The precision of Circuit/CompositeGate, should be one of [single, double]
        """
        assert precision in ["single", "double"], "Circuit's precision should be one of [double, single]"
        self._precision = precision

    def get_variable_shape(self):
        for gate, _, _ in self._gates:
            if gate.variables == 0:
                continue
            for i in range(gate.params):
                if isinstance(gate.pargs[i], Variable):
                    return gate.pargs[i].origin_shape

    def get_variables(self):
        shape = self.get_variable_shape()
        pargs = np.zeros(shape=shape, dtype=np.float64)
        grads = np.zeros(shape=shape, dtype=np.float64)

        remain_training_gates = self.count_training_gate()
        for gate, _, _ in self._gates:
            if remain_training_gates == 0:
                break
            if gate.variables == 0:
                continue
            remain_training_gates -= 1
            for i in range(gate.params):
                if isinstance(gate.pargs[i], Variable):
                    index = gate.pargs[i].index
                    pargs[index] = gate.pargs[i].pargs
                    grads[index] = gate.pargs[i].grads
        return Variable(pargs=pargs, grads=grads)

    def update(self, variables):
        assert variables.shape == self.get_variable_shape()
        remain_training_gates = self.count_training_gate()
        for gate, _, _ in self._gates:
            if remain_training_gates == 0:
                return
            if gate.variables == 0:
                continue
            remain_training_gates -= 1
            for i in range(gate.params):
                if isinstance(gate.pargs[i], Variable):
                    index = gate.pargs[i].index
                    gate.pargs[i].pargs = variables.pargs[index]
                    gate.pargs[i].grads = variables.grads[index]

    def show_detail(self):
        """
        Print the list of gates in the Circuit/CompositeGate
        """
        for g in self.flatten_gates():
            print(g.type, g.cargs, g.targs, g.pargs)

    def draw(self, method: str = 'matp_auto', filename: str = None, flatten: bool = False):
        """Draw the figure of circuit.

        Args:
            method(str): the method to draw the circuit
                matp_inline: Show the figure interactively but do not save it to file.
                matp_file: Save the figure to file but do not show it interactively.
                matp_auto: Automatically select inline or file mode according to matplotlib backend.
                matp_silent: Return the drawn figure without saving or showing.
                command : command
            filename(str): the output filename without file extensions, default to None.
                If filename is None, it will using matlibplot.show() except matlibplot.backend
                is agg, it will output jpg file named circuit's name.
            flatten(bool): Whether draw the Circuit with CompositeGate or Decomposite it.

        Returns:
            If method is 'matp_silent', a matplotlib Figure is returned. Note that that figure is created in matplotlib
            Object Oriented interface, which means it must be display with IPython.display.

        Examples:
            >>> from IPython.display import display
            >>> circ = Circuit(5)
            >>> circ.random_append()
            >>> silent_fig = circ.draw(method="matp_silent")
            >>> display(silent_fig)

            >>> from IPython.display import display
            >>> compositegate = CompositeGate()
            >>> cx_gate=CX & [1,3]
            >>> u2_gate= U2(1, 0)
            >>> H| compositegate(1)
            >>> cx_gate | compositegate
            >>> u2_gate | compositegate(1)
            >>> silent_fig = compositegate.draw(method="matp_silent")
            >>> display(silent_fig)
        """
        from QuICT.tools.drawer import PhotoDrawer, TextDrawing
        import matplotlib

        if method.startswith('matp'):
            if filename is not None:
                if '.' not in filename:
                    filename += '.jpg'

            photo_drawer = PhotoDrawer()
            if method == 'matp_auto':
                save_file = matplotlib.get_backend() == 'agg'
                show_inline = matplotlib.get_backend() != 'agg'
            elif method == 'matp_file':
                save_file = True
                show_inline = False
            elif method == 'matp_inline':
                save_file = False
                show_inline = True
            elif method == 'matp_silent':
                save_file = False
                show_inline = False
            else:
                raise ValueError(
                    "Circuit.draw.matp_method", "[matp_auto, matp_file, matp_inline, matp_silent]", method
                )

            silent = (not show_inline) and (not save_file)
            photo_drawer.run(circuit=self, filename=filename, save_file=save_file, flatten=flatten)

            if show_inline:
                from IPython.display import display
                display(photo_drawer.figure)
            elif silent:
                return photo_drawer.figure

        elif method == 'command':
            gates = self.flatten_gates() if flatten else self.gates
            text_drawer = TextDrawing(self._qubits, gates)
            if filename is None:
                print(text_drawer.single_string())
                return
            elif '.' not in filename:
                filename += '.txt'

            text_drawer.dump(filename)
        else:
            raise ValueError(
                "Circuit.draw.method", "[matp_auto, matp_file, matp_inline, matp_silent, command]", method
            )
