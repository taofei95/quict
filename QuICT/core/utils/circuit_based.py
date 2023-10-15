#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2022/1/15 10:31
# @Author  : Han Yu, Li Kaiqi
# @File    : _circuit_computing.py
from enum import Enum
import numpy as np
from typing import Union

from .id_generator import unique_id_generator
from .gate_type import GateType
from .variable import Variable
from .circuit_gate import CircuitGates


class CircuitBased(object):
    """ Based Class for Circuit and Composite Gate. """

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str):
        if name is None:
            self._name = "QC_" + unique_id_generator()
        else:
            self._name = name

    @property
    def precision(self) -> str:
        return self._precision

    @precision.setter
    def precision(self, precision: str):
        assert precision in ['single', 'double'], ValueError("Wrong precision. Should be one of [single, double]")
        self._precision = precision

    def __init__(self, name: str, qubits: int = 0, precision: str = "double"):
        """
        Args:
            name (str): The name of current Quantum Circuit
        """
        self.name = name
        self.precision = precision

        self._gates = CircuitGates()
        self._qubits = qubits
        self._pointer = None

    ####################################################################
    ############         Circuit's Gates Function           ############
    ####################################################################
    @property
    def gates(self) -> list:
        """ Return the list of BasicGate/CompositeGate/Operator in the current circuit. \n
        *Warning*: this is slowly due to the copy of gates, you can use self.fast_gates to
        avoid gate copy.
        """
        gate_list = []
        for gate in self._gates.gates:
            if type(gate).__name__ == "CompositeGate":
                gate_list.append(gate.copy())
            else:
                gate_list.append(gate.gate.copy() & gate.indexes)

        return gate_list

    @property
    def fast_gates(self) -> list:
        """ Return the list of Tuple(Union[BasicGate, CompositeGate], indexes) in the current circuit. """
        gate_list = []
        for gate in self._gates.gates:
            if type(gate).__name__ == "CompositeGate":
                gate_list.append((gate, gate.qubits))
            else:
                gate_list.append((gate.gate, gate.indexes))

        return gate_list

    def decomposition_gates(self) -> list:
        """ Decomposition the CompositeGate or BasicGate which has build_gate function.

        Returns:
            list: The list of BasicGate
        """
        return self._gates.decomposition(True)

    def flatten_gates(self) -> list:
        """ Get the list of Quantum Gates, flat all the CompositeGate in Circuit. """
        return [gate.copy() & qidx for gate, qidx in self._gates.tree_search()]

    def decomposition(self):
        """ Decomposition the CompositeGate and BasicGate in current Circuit. """
        self._gates.decomposition()

    def flatten(self):
        """ Flat all CompositeGate in current Circuit. """
        self._gates.flatten()

    def get_target_gates(self, qubits: list, depth: int) -> list:
        """ Get the list of Quantum Gates in target qubits and depth.

        Args:
            qubits (list): The target qubit indexes.
            depth (int): The target depth.
        """
        return self._gates.tree_search(qubits, depth)

    def pop(self, index: int = -1):
        """ Pop the BasicGate/Operator/CompositeGate from current Quantum Circuit.

        Args:
            index (int, optional): The target gate's index. Defaults to 0.
        """
        if index < 0:
            index = self.gate_length() + index

        assert index >= 0 and index < self.gate_length()
        return self._gates.pop(index)

    def insert(self, gate, qubits: list = None, depth: int = -1):
        """ Insert a Quantum Gate into current Circuit.

        Args:
            gate (Union[BasicGate, CompositeGate]): The quantum gate want to insert
            qubits (list[int]): The target qubit indexes.
            depth (int): The index of insert position.
        """
        if qubits is None:
            qubits = gate.qubits if type(gate).__name__ == "CompositeGate" else gate.cargs + gate.targs

        if len(qubits) == 0:
            raise ValueError("Gate need qubit indexes to insert into Composite Gate.")

        self._qubit_indexes_validation(qubits)
        if type(gate).__name__ == "CompositeGate":
            self._gates.insert_cgate(gate, qubits, depth)
        else:
            self._gates.insert_gate(gate, qubits, depth)

    def split(self, qubits: list = None, depth: Union[int, list] = None):
        if qubits is None and depth is None:
            raise KeyError("Split must assign at least one of qubits and depth.")



    ####################################################################
    ############           Circuit's Properties             ############
    ####################################################################
    def size(self) -> int:
        """ the number of BasicGate/Operator in the Circuit.

        Returns:
            int: the number of gates in circuit
        """
        return self._gates.size

    def gate_length(self) -> int:
        """ The number of CompositeGate and BasicGate in Circuit. """
        return self._gates.length

    def width(self):
        """ The number of qubits in Circuit.

        Returns:
            int: the number of qubits in circuit
        """
        return len(self.qubits)

    def depth(self) -> int:
        """ the depth of the circuit.

        Returns:
            int: the depth
        """
        return self._gates.depth()

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
        return self._gates.siq_gates_count

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

    def show_detail(self):
        """
        Print the list of gates in the Circuit/CompositeGate
        """
        for g in self.flatten_gates():
            print(g.type, g.cargs, g.targs, g.pargs)

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

    ####################################################################
    ############           Circuit's Utilities              ############
    ####################################################################
    def _qubit_indexes_validation(self, indexes: list):
        """ Validate the qubit indexes.

        Args:
            indexes (list): The given qubit indexes.
        """
        # Indexes' type check
        if not isinstance(indexes, list):
            raise TypeError(
                f"Qubit indexes should be one of int/list[int]/Qubit/Qureg not {type(indexes)}."
            )
        for idx in indexes:
            assert idx >= 0 and isinstance(idx, (int, np.int32, np.int64)), \
                "The qubit indexes should be integer and greater than zero."

        # Repeat indexes check
        if len(indexes) != len(set(indexes)):
            raise ValueError(
                "The qubit indexes cannot contain the repeatted index."
            )

        # Qubit's indexes max/min limitation check
        min_idx, max_idx = min(indexes), max(indexes)
        if min_idx < 0:
            raise ValueError("The qubit indexes should >= 0.")

        if self._qubits != 0:
            assert max_idx < self.width(), ValueError("The max of qubit indexes cannot exceed the width of Circuit.")
            assert len(indexes) <= self.width(), \
                ValueError("The number of qubit index cannot exceed the width of Circuit.")

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
        for gate, targs in self._gates.tree_search():
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
            text_drawer = TextDrawing(self.qubits, gates)
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
    Arithmetic = "Arithmetic"
    Misc = "Misc"
