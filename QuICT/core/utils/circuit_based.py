#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/15 10:31
# @Author  : Han Yu, Li Kaiqi
# @File    : _circuit_computing.py
from enum import Enum
import numpy as np

from .gate_type import GateType


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

    @property
    def gates(self) -> list:
        """ Return the list of BasicGate/CompositeGate/Operator in the current circuit. \n
        *Warning*: this is slowly due to the copy of gates, you can use self.fast_gates to
        get list of tuple(gate, qubit_indexes, size) for further using.
        """
        combined_gates = [gate.copy() & targs for gate, targs, _ in self._gates]

        return combined_gates

    @property
    def fast_gates(self) -> list:
        """ Return the list of tuple(gate, qubit_indexes, size) in the current circuit. """
        return self._gates

    def flatten_gates(self, decomposition: bool = False) -> list:
        """ Get the list of Quantum Gates with decompose the CompositeGate. 

        Args:
            decomposition (bool, optional): Whether call build_gate for Quantum Gates. Defaults to False.

        Returns:
            List[BasicGate]: The list of BasicGate/Operator.
        """
        flatten_gates = self.gate_decomposition(self_flatten=False, decomposition=decomposition)

        return [gate & qidxes for gate, qidxes, _ in flatten_gates]

    def __init__(self, name: str):
        """
        Args:
            name (str): The name of current Quantum Circuit
        """
        self._name = name
        self._gates = []
        self._pointer = None
        self._precision = "double"

    def size(self) -> int:
        """ the number of gates in the circuit/CompositeGate, the operators are not count.

        Returns:
            int: the number of gates in circuit
        """
        tsize = 0
        for _, _, size in self._gates:
            tsize += size

        return tsize

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
        depth = np.zeros(self.width(), dtype=int)
        for _, targs, _ in self.gate_decomposition(False, False):
            depth[targs] = np.max(depth[targs]) + 1

        return np.max(depth)

    def count_2qubit_gate(self) -> int:
        """ the number of the two qubit gates in the Circuit/CompositeGate

        Returns:
            int: the number of the two qubit gates
        """
        count = 0
        for gate, _, size in self._gates:
            if size == 0:
                continue

            if size > 1 or hasattr(gate, "count_2qubit_gate"):
                count += gate.count_2qubit_gate()
                continue

            if gate.controls + gate.targets == 2:
                count += 1

        return count

    def count_1qubit_gate(self) -> int:
        """ the number of the one qubit gates in the Circuit/CompositeGate

        Returns:
            int: the number of the one qubit gates
        """
        count = 0
        for gate, _, size in self._gates:
            if size == 0:
                continue

            if size > 1 or hasattr(gate, "count_1qubit_gate"):
                count += gate.count_1qubit_gate()
                continue

            if gate.controls + gate.targets == 1:
                count += 1

        return count

    def count_gate_by_gatetype(self, gate_type: GateType) -> int:
        """ the number of the target Quantum Gate in the Circuit/CompositeGate

        Args:
            gateType(GateType): the type of gates to be count

        Returns:
            int: the number of the gates
        """
        count = 0
        for gate, _, size in self._gates:
            if size == 0:
                continue

            if size > 1 or hasattr(gate, "count_gate_by_gatetype"):
                count += gate.count_gate_by_gatetype(gate_type)
                continue

            if gate.type == gate_type:
                count += 1

        return count

    def __str__(self):
        circuit_info = {
            "name": self.name,
            "width": self.width(),
            "size": self.size(),
            "depth": self.depth(),
            "1-qubit gates": self.count_1qubit_gate(),
            "2-qubit gates": self.count_2qubit_gate()
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
        qasm_string = ""
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
            with open(output_file, 'w+') as of:
                of.write(qasm_string)

        return qasm_string

    def set_precision(self, precision: str):
        """ Set precision for Cicuit/CompositeGate

        Args:
            precision(str): The precision of Circuit/CompositeGate, should be one of [single, double]
        """
        assert precision in ["single", "double"], "Circuit's precision should be one of [double, single]"
        self._precision = precision

    def gate_decomposition(self, self_flatten: bool = True, decomposition: bool = True) -> list:
        """ Decomposition the CompositeGate or BasicGate which has build_gate function.

        Args:
            self_flatten (bool, optional): Whether change the gates in current Circuit. Defaults to True.
            decomposition (bool, optional): Whether call build_gate for BasicGates. Defaults to True.

        Returns:
            list: The list of (gate, qubit indexes, size)
        """
        decomp_gates = []
        for gate, qidxes, size in self._gates:
            if size > 1 or hasattr(gate, "gate_decomposition"):
                temp_gate = gate.copy() & qidxes
                decomp_gates += temp_gate.gate_decomposition(False, False)
            else:
                decomp_gates.append((gate, qidxes, size))

        if decomposition:
            temp_decomp_gates = []
            for gate, qidxes, size in decomp_gates:
                cgate = gate.build_gate(qidxes)
                if cgate is not None:
                    temp_decomp_gates += cgate._gates
                else:
                    temp_decomp_gates.append((gate, qidxes, size))

            decomp_gates = temp_decomp_gates[:]

        if self_flatten:
            self._gates = decomp_gates

        return decomp_gates

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

        if flatten:
            self.gate_decomposition(decomposition=False)

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
            photo_drawer.run(circuit=self, filename=filename, save_file=save_file)

            if show_inline:
                from IPython.display import display
                display(photo_drawer.figure)
            elif silent:
                return photo_drawer.figure

        elif method == 'command':
            text_drawer = TextDrawing(self._qubits, self.gates)
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


class CircuitMode(Enum):
    Clifford = "Clifford"
    CliffordRz = "CliffordRz"
    Arithmetic = 'Arithmetic'
    Misc = "Misc"
