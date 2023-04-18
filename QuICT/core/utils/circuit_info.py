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
        get list of tuple(gate, qidxes, size) for further using.
        """
        combined_gates = [gate.copy() & targs for gate, targs, _ in self._gates]

        return combined_gates

    @property
    def fast_gates(self) -> list:
        """ Return the list of tuple(gates' info) in the current circuit. it contains the gate,
        the qubit indexes and the gate's size."""
        return self._gates

    def flatten_gates(self, decomposition: bool = False) -> list:
        """ Return the list of BasicGate/Operator. """
        flatten_gates = []
        for gate, qidxes, _ in self._gates:
            gate = gate.copy() & qidxes
            if hasattr(gate, "flatten_gates"):
                flatten_gates.extend(gate.flatten_gates(decomposition))
            else:
                if decomposition:
                    cgate = gate.build_gate()
                    if cgate is not None:
                        flatten_gates.extend(cgate.gates)
                        continue

                flatten_gates.append(gate)

        return flatten_gates

    def __init__(self, name: str):
        self._name = name
        self._gates = []
        self._pointer = None
        self._precision = "double"

    def size(self) -> int:
        """ the number of gates in the circuit/CompositeGate

        Returns:
            int: the number of gates in circuit
        """
        tsize = 0
        for _, _, size in self._gates:
            tsize += size

        return tsize

    def width(self):
        """ the number of qubits in circuit

        Returns:
            int: the number of qubits in circuit
        """
        return len(self._qubits)

    def depth(self, depth_per_qubits: bool = False) -> int:
        """ the depth of the circuit.

        Returns:
            int: the depth
        """
        depth = np.zeros(self.width(), dtype=int)
        for gate, targs, _ in self._gates:
            if hasattr(gate, "depth"):
                gdepth = gate.depth(True)
                for i, targ in enumerate(targs):
                    depth[targ] += gdepth[i]
            else:
                depth[targs] = np.max(depth[targs]) + 1

        return np.max(depth) if not depth_per_qubits else depth

    def count_2qubit_gate(self) -> int:
        """ the number of the two qubit gates in the circuit/CompositeGate

        Returns:
            int: the number of the two qubit gates
        """
        count = 0
        for gate, _, size in self._gates:
            if size > 1 or hasattr(gate, "count_2qubit_gate"):
                count += gate.count_2qubit_gate()
                continue

            if gate.controls + gate.targets == 2:
                count += 1

        return count

    def count_1qubit_gate(self) -> int:
        """ the number of the one qubit gates in the circuit/CompositeGate

        Returns:
            int: the number of the one qubit gates
        """
        count = 0
        for gate, _, size in self._gates:
            if size > 1 or hasattr(gate, "count_1qubit_gate"):
                count += gate.count_1qubit_gate()
                continue

            if gate.controls + gate.targets == 1:
                count += 1

        return count

    def count_gate_by_gatetype(self, gate_type: GateType) -> int:
        """ the number of the gates which are some type in the circuit/CompositeGate

        Args:
            gateType(GateType): the type of gates to be count

        Returns:
            int: the number of the gates which are some type
        """
        count = 0
        for gate, _, size in self._gates:
            if size > 1 or hasattr(gate, "count_gate_by_gatetype"):
                count += gate.count_gate_by_gatetype(gate_type)
            elif gate.type == gate_type:
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
        """ The qasm of current CompositeGate/Circuit.

        Args:
            output_file (str): The output qasm file's name or path

        Returns:
            str: The string of Circuit/CompositeGate's qasm.
        """
        qasm_string = ""
        qreg = self.width()
        creg = min(self.count_gate_by_gatetype(GateType.measure), qreg)
        if creg == 0:
            creg = qreg

        qasm_string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        qasm_string += f"qreg q[{qreg}];\n"
        qasm_string += f"creg c[{creg}];\n"

        cbits = 0
        for gate, targs, size in self._gates:
            if size > 1 or hasattr(gate, "qasm_gates_only"):
                qasm_string += gate.qasm_gates_only(creg, cbits, targs)
                continue

            if gate.qasm_name == "measure":
                qasm_string += f"measure q[{targs}] -> c[{cbits}];\n"
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

    def gate_decomposition(self, self_flatten: bool = True) -> list:
        decomp_gates = []
        for gate, qidxes, size in self._gates:
            if size > 1 or hasattr(gate, "gate_decomposition"):
                decomp_gates += gate.gate_decomposition()
                continue

            cgate = gate.build_gate()
            if cgate is not None:
                cgate & qidxes
                decomp_gates += cgate._gates
                continue

            decomp_gates.append((gate, qidxes, size))

        if not self_flatten:
            return decomp_gates
        else:
            self._gates = decomp_gates
            return self._gates

    def draw(self, method: str = 'matp_auto', filename: str = None):
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
            get_figure(bool): Whether to return the figure object of matplotlib.

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
