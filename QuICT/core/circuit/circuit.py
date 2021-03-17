#!/usr/bin/env python
# -*- coding:utf8 -*-
# @TIME    : 2020/2/10 9:41
# @Author  : Han Yu
# @File    : _circuit.py

import copy

from QuICT.core.qubit import Qubit, Qureg
from QuICT.core.exception import TypeException, ConstException, IndexLimitException, IndexDuplicateException
from .circuit_computing import *

# global circuit id count
circuit_id = 0

class Circuit(object):
    """ Implement a quantum circuit

    Circuit is the core part of the framework.

    Attributes:
        id(int): the unique identity code of a circuit, which is generated globally.
        const_lock(bool): a simple lock that ensures the circuit does not change when running some algorithm.
        name(str): the name of the circuit
        qubits(Qureg): the qureg formed by all qubits of the circuit
        gates(list<BasicGate>): all gates attached to the circuit
        topology(list<tuple<int, int>>):
            The topology of the circuit. When the topology list is empty, it will be seemed as fully connected.
        fidelity(float): the fidelity of the circuit

    Private Attributes:
        __idmap(dictionary): the map from qubit's id to its index in the circuit
        __queue_gates(list<BasicGate>): the gates haven't be flushed

    """

    @property
    def id(self) -> int:
        return self.__id

    @id.setter
    def id(self, id):
        if not self.const_lock:
            self.__id = id
        else:
            raise ConstException(self)

    @property
    def const_lock(self) -> bool:
        return self.__const_lock

    @const_lock.setter
    def const_lock(self, const_lock):
        self.__const_lock = const_lock

    @property
    def name(self) -> int:
        return self.__name

    @name.setter
    def name(self, name):
        if not self.const_lock:
            self.__name = name
        else:
            raise ConstException(self)

    @property
    def qubits(self) -> Qureg:
        return self.__qubits

    @qubits.setter
    def qubits(self, qubits):
        if not self.const_lock:
            self.__qubits = qubits
        else:
            raise ConstException(self)

    @property
    def gates(self) -> list:
        return self.__gates

    @gates.setter
    def gates(self, gates):
        if not self.const_lock:
            self.__gates = gates
        else:
            raise ConstException(self)

    @property
    def topology(self) -> list:
        return self.__topology

    @topology.setter
    def topology(self, topology):
        if not self.const_lock:
            self.__topology = topology
        else:
            raise ConstException(self)

    @property
    def fidelity(self) -> float:
        return self.__fidelity

    @fidelity.setter
    def fidelity(self, fidelity):
        if fidelity is None:
            self.__fidelity = None
            return
        if not isinstance(fidelity, float) or fidelity < 0 or fidelity > 1.0:
            raise Exception("fidelity should be in [0, 1]")
        if not self.const_lock:
            self.__fidelity = fidelity
        else:
            raise ConstException(self)

    # life cycle
    def __init__(self, wires):
        """ generator a circuit

        Args:
            wires(int): the number of qubits in the circuit
        """
        self.const_lock = False
        global circuit_id
        self.id = circuit_id
        self.name = "circuit" + str(self.id)
        circuit_id = circuit_id + 1
        self.__idmap = {}
        self.qubits = Qureg()
        for idx in range(wires):
            qubit = Qubit(self)
            self.qubits.append(qubit)
            self.__idmap[qubit.id] = idx
        self.gates = []
        self.__queue_gates = []
        self.topology = []
        self.fidelity = None

    def __del__(self):
        """ release the memory

        """
        for qubit in self.qubits:
            qubit.qState_clear()
        self.const_lock = False
        self.gates = None
        self.__queue_gates = None
        self.qubits = None
        self.topology = None

    # Attributes of the circuit
    def circuit_width(self):
        """ the number of qubits in circuit

        Returns:
            int: the number of qubits in circuit
        """
        return len(self.qubits)

    def circuit_size(self):
        """ the size of the circuit

        Returns:
            int: the number of gates in circuit
        """
        return len(self.gates)

    def circuit_count_2qubit(self):
        """ the number of the two qubit gates in the circuit

        Returns:
            int: the number of the two qubit gates in the circuit
        """
        count = 0
        for gate in self.gates:
            if gate.controls + gate.targets == 2:
                count += 1
        return count

    def circuit_count_1qubit(self):
        """ the number of the one qubit gates in the circuit

        Returns:
            int: the number of the one qubit gates in the circuit
        """
        count = 0
        for gate in self.gates:
            if gate.controls + gate.targets == 1:
                count += 1
        return count

    def circuit_count_gateType(self, gateType):
        """ the number of the gates which are some type in the circuit

        Args:
            gateType(GateType): the type of gates to be count

        Returns:
            int: the number of the gates which are some type in the circuit
        """
        count = 0
        for gate in self.gates:
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
        for gate in self.gates:
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
        string = 'OPENQASM 2.0;\ninclude "qelib1.inc";\n'
        cbits = 0
        for gate in self.gates:
            if gate.qasm_name == "measure":
                cbits += 1
        string += f"qreg q[{self.circuit_width()}];\n"
        if cbits != 0:
            string += f"creg c[{cbits}];\n"
        cbits = 0
        for gate in self.gates:
            if gate.qasm_name == "measure":
                string += f"measure q[{gate.targ}] -> c[{cbits}];\n"
                cbits += 1
            else:
                qasm = gate.qasm()
                if qasm == "error":
                    print("the circuit cannot be transformed to a valid describe in OpenQASM 2.0")
                    gate.print_info()
                    return "error"
                string += gate.qasm()
        return string

    # slice method of circuit
    def __call__(self, indexes: object) -> Qureg:
        """ get a smaller qureg from this circuit

        Args:
            indexes: the indexes passed in, it can have follow form:
                1) int
                2) list<int>
                3) tuple<int>
        Returns:
            Qureg: the qureg correspond to the indexes
        Exceptions:
            IndexDuplicateException: the range of indexes is error.
            TypeException: the type of indexes is error.
        """
        # int
        if isinstance(indexes, int):
            if indexes < 0 or indexes >= len(self.qubits):
                raise IndexLimitException(len(self.qubits), indexes)
            return Qureg(self.qubits[indexes])

        # tuple
        if isinstance(indexes, tuple):
            indexes = list(indexes)

        # list
        if isinstance(indexes, list):
            if len(indexes) != len(set(indexes)):
                raise IndexDuplicateException(indexes)
            qureg = Qureg()
            for element in indexes:
                if not isinstance(element, int):
                    raise TypeException("int", element)
                if element < 0 or element >= len(self.qubits):
                    raise IndexLimitException(len(self.qubits), element)
                qureg.append(self.qubits[element])
            return qureg

        raise TypeException("int or list or tuple", indexes)

    def __getitem__(self, item):
        """ to fit the slice operator, overloaded this function.

        get a smaller qureg/qubit from this circuit

        Args:
            item(int/slice): slice passed in.
        Return:
            Qubit/Qureg: the result or slice
        """

        if isinstance(item, int):
            return self.qubits[item]
        elif isinstance(item, slice):
            qureg_list = self.qubits[item]
            return qureg_list

    def __or__(self, targets):
        """deal the operator '|'

        Use the syntax "circuit | circuit" or "circuit | qureg" or "circuit | qubit"
        to add the gate of circuit into the circuit/qureg/qubit

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
        for gate in self.gates:
            gate | targets

    # append gate methods
    def append(self, gate, qureg = None):
        """ add a gate to the circuit

        Args:
            gate(BasicGate): the gate to be added to the circuit
            qureg(Qureg/Qubit/None): the Qureg/Qubit gate added to.
                                    if Qureg is None, use the targs and cargs in gate
        """
        if qureg is None:
            qureg_list = [] if gate.controls == 0 else copy.deepcopy(gate.cargs)
            qureg_list.extend(gate.targs)
            qureg = self(qureg_list)
        self._add_gate(gate, qureg)

    def extend(self, gates):
        """ add gates to the circuit

        Args:
            gates(list<BasicGate>): the gate to be added to the circuit
        """
        for gate in gates:
            self.append(gate)

    def sub_circuit(self, targets, start = 0, max_size = -1, local = False, remove = False):
        """ get a sub circuit

        Args:
            targets(int/list<int>/tuple<int>/slice): target qubits indexes.
            start(int): the start gate's index, default 0
            max_size(int): max size of the sub circuit, default -1 without limit
            local(bool): whether the slice will stop when meeting an non-commutative gate, default False
            remove(bool): whether deleting the slice gates from origin circuit, default False
        Return:
            Circuit: the sub circuit

        """
        circuit_size   = self.circuit_size()
        circuit_width  = self.circuit_width()
        if isinstance(targets, slice):
            targets = [i for i in range(circuit_size)][targets]
        targets = list(targets)

        # the mapping from circuit's index to sub-circuit's index
        targets_mapping = [0] * circuit_width
        index = 0
        for target in targets:
            if target < 0 or target >= circuit_width:
                raise Exception(f'list index out of range')
            targets_mapping[target] = index
            index += 1

        circuit = Circuit(len(targets))
        set_targets = set(targets)
        new_gates = []
        compare_gates = []

        for gate_index in range(start, circuit_size):
            gate = self.gates[gate_index]
            affectArgs = gate.affectArgs
            set_affectArgs = set(affectArgs)
            # if gate acts on given qubits
            if set_affectArgs <= set_targets:
                compare_gates.append(gate)
                gate | circuit(targets)
                max_size -= 1
            elif len(set_affectArgs.intersection(set_targets)) == 0:
                new_gates.append(gate)
            else:
                if not local:
                    new_gates.append(gate)
                else:
                    communitive = True
                    for goal in compare_gates:
                        if not gate.communitive(goal):
                            communitive = False
                            break
                    if communitive:
                        new_gates.append(gate)
                    else:
                        break
            if max_size <= 0:
                break

        if remove:
            for qubit in self.qubits:
                qubit.qState_clear()
            self.set_exec_gates(new_gates)

        return circuit

    def _add_gate(self, gate, qureg):
        """ add a gate into some qureg

        Args:
            gate(BasicGate)
            qureg(Qureg/list<Qubit>)
        """
        self.gates.append(gate)
        self.__queue_gates.append(gate)
        gate.cargs = [self.__idmap[qureg[idx].id] for idx in range(gate.controls)]
        gate.targs = [self.__idmap[qureg[idx].id] for idx in range(gate.controls, gate.controls + gate.targets)]

    # add topology methods
    def add_topology(self, topology):
        """ public API to add directed edges in topology to the circuit

        Args:
            topology(list<tuple<qureg/qubit/int>>): a list of directed edges
        """
        if isinstance(topology, list):
            for item in topology:
                self._inner_add_topology(item)
        else:
            self._inner_add_topology(topology)

    def add_topology_complete(self, qureg : Qureg):
        """ add directed edges to make subgraph formed by qureg passed in fully connected

        Args:
            qureg(Qureg): the qureg to be fully connected in topology
        """
        ids = []
        for qubit in qureg:
            ids.append(self.__idmap[qubit.id])
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                self._inner_add_topology((i, j))

    def _inner_add_topology(self, topology):
        """ add a directed edge in topology to the circuit

        Args:
            topology(tuple<qureg/qubit/int>): the two qubits with is connected directly

        Raises:
            TypeException:
        """
        if not isinstance(topology, tuple):
            raise TypeException("tuple<qureg(len = 1)/qubit/int> or list<tuple<qureg(len = 1)/qubit/int>>", topology)
        if len(topology) != 2:
            raise Exception("there should be two element")
        item1 = topology[0]
        item2 = topology[1]
        if isinstance(item1, Qureg):
            if len(item1) != 1:
                item1 = item1[0]
        if isinstance(item1, Qubit):
            item1 = self.__idmap[item1]
        if not isinstance(item1, int):
            raise TypeException("tuple<qureg(len = 1)/qubit/int> or list<tuple<qureg(len = 1)/qubit/int>>", topology)
        if isinstance(item2, Qureg):
            if len(item2) != 1:
                item2 = item2[0]
        if isinstance(item2, Qubit):
            item2 = self.__idmap[item2]
        if not isinstance(item2, int):
            raise TypeException("tuple<qureg(len = 1)/qubit/int> or list<tuple<qureg(len = 1)/qubit/int>>", topology)
        self.topology.append((item1, item2))

    # exec method
    def exec(self):
        """ calculate the gates applied to the circuit, change the qStates' values in the circuit

        It may cost a lot of time, the computational process is coded by
        C++ with intel tbb parallel library

        """
        for gate in self.__queue_gates:
            gate.exec(self)
        self.__queue_gates = []

    def reset(self):
        """ reset all qubits in the circuit with gates unchanged

        """
        for qubit in self.qubits:
            qubit.qState_clear()
        self.__queue_gates = copy.deepcopy(self.gates)

    def clear(self):
        """ clear the circuit

        """
        self.reset()
        self.gates = []
        self.__queue_gates = []

    def set_exec_gates(self, gates):
        """ set circuit's gates

        Args:
            gates(list<BasicGate>)
        """
        self.gates = gates.copy()
        self.__queue_gates = gates.copy()

    # set initial state
    def assign_initial_random(self):
        """ remove all gates and set all qubits into a qState with random amplitude

        """
        self.clear()
        self.qubits.force_assign_random()

    def assign_initial_zeros(self):
        """ remove all gates and set all qubits into a qState with state 0

        """
        self.clear()
        self.qubits.force_assign_zeros()

    def force_copy(self, other, force_copy = None):
        """ remove all gates and copy another circuits' qubits state to the qureg in this circuit

        Args:
            other(Circuit): the copy goal
            force_copy: the indexes of qureg to be pasted in the circuit,
                        if want cover whole circuit, leave it to be None
        :return:
        """
        self.clear()
        if force_copy is None:
            force_copy = [i for i in range(len(self.qubits))]
        self.qubits.force_copy(other.qubits[0].qState, force_copy)

    def exec_release(self):
        """ calculate the gates applied to the circuit and remove the gates in the circuits

        when call the function "exec", the gates in the list "self.gates" won't be removed because
        the integrity of the information of circuit should be ensured.
        But sometimes, user may release the memory by removing them.

        """
        self.exec()
        self.gates = []

    # display information of the circuit
    def print_infomation(self):
        print("-------------------")
        print(f"number of bits:{self.circuit_width()}")
        for gate in self.gates:
            gate.print_info()
        print("-------------------")

    def draw_photo(self, filename = None, show_depth = True):
        """ draw the photo of circuit in the run directory

        Args:
            filename(str): the output filename without file extensions,
                           default to be the name of the circuit
            show_depth: whether to show a red frame for gates in the same layer
                        (Note that some gates in the same layer cannot be represented
                        in the row in the photo)
        """
        from QuICT.tools.drawer import PhotoDrawerModel
        if filename is None:
            filename = str(self.id) + '.jpg'
        elif '.' not in filename:
            filename += '.jpg'
        PhotoDrawer = PhotoDrawerModel()
        PhotoDrawer.run(self, filename, show_depth)

    # computation method of the circuit
    def partial_prob(self, indexes):
        """ calculate the probabilities of the measure result of partial qureg in circuit

        Note that the function "flush" will be called before calculating
        this function is a cheat function, which do not change the state of the qureg.

        Args:
            indexes(list<int>): the indexes of the partial qureg.

        Returns:
            list<float>: the probabilities of the measure result, the memory mode is LittleEndian.

        """
        return inner_partial_prob(self, indexes)

    def index_for_qubit(self, qubit, ancilla = None) -> int:
        """ find the index of qubit in this circuit

        the index ignored the ancilla qubit

        Args:
            qubit(Qubit): the qubit need to be indexed.
            ancilla(list<Qubit>): the ancillary qubit

        Returns:
            int: the index of the qubit.

        Raises:
            Exception: the qubit is not in the circuit
        """
        if not isinstance(qubit, Qubit):
            raise TypeException("Qubit", now=qubit)
        if ancilla is None:
            ancilla = []
        enterspace = 0
        for i in range(len(self.qubits)):
            if i in ancilla:
                enterspace += 1
            elif self.qubits[i].id == qubit.id:
                return i - enterspace
        raise Exception("the qubit is not in the circuit or it is an ancillary qubit.")

    def random_append(self, rand_size = 10, typeList = None):
        """ add some random gate to the circuit
        Args:
            rand_size(int): the number of the gate added to the circuit
            typeList(list<GateType>): the type of gate, default contains CX、ID、Rz、CY、CRz、CH
        """
        inner_random_append(self, rand_size, typeList)

    def matrix_product_to_circuit(self, gate) -> np.ndarray:
        """ extend a gate's matrix in the all circuit unitary linear space

        gate's matrix tensor products some identity matrix.

        Args:
            gate(BasicGate): the gate to be extended.

        """
        return inner_matrix_product_to_circuit(self, gate)
