
class CircuitCostMeasure(object):
    """
    Measure of gate cost in quantum circuit.
    """

    def __init__(self, method='static', cost_dict=None):
        """
        Args:
            method(str): Method of cost measure. Currently only 'static' is supported.
            cost_dict(Dict[GateType, int]): Cost of each gate type (used in static method).
                It a gate type is not specified, its cost is treated as 0.
        """
        self.method = method
        if self.method is 'static':
            if cost_dict is None:
                self.cost_dict = self.NISQ_GATE_COST
            else:
                self.cost_dict = cost_dict

    def __getitem__(self, gate_type):
        """
        Get cost of a gate type. Subscript can be a GateType or string of a gate type.

        Args:
            gate_type(GateType/str): Gate type

        Returns:
            int: Cost of the gate type
        """
        if isinstance(gate_type, str):
            gate_type = GateType(gate_type)

        if gate_type in self.cost_dict:
            return self.cost_dict[gate_type]
        else:
            return 0

    def _cost_static(self, circuit):
        """
        Calculate cost of a circuit based on static gate cost.
        """
        if isinstance(circuit, Iterable):
            return sum(self[n.gate.type] for n in circuit)
        else:
            return sum(self[n.gate.type] for n in circuit.gates)

    def _gate_fidelity(self, circuit, gate):
        return 1

    def _cost2(self, circuit: Circuit):
        cost = 0
        for g in circuit.gates:
            f = self._gate_fidelity(circuit, g)
            cost += -log(f)
        return cost

    def _cost_dynamic(self, circuit: Circuit):
        """
        Calculate cost of a circuit based on fidelity.
        """
        qubit_f = np.array([q.fidelity for q in circuit.qubits])
        for g in circuit.gates:
            g: BasicGate
            for qubit_ in g.cargs + g.targs:
                qubit_f[qubit_] = cos(acos(qubit_f[qubit_]) + acos(g.fidelity))
        return np.prod(qubit_f)

    def cost(self, circuit):
        """
        Calculate cost of a circuit.

        Args:
            circuit(Circuit): Circuit to be measured

        Returns:
            int: Cost of the circuit
        """
        if self.method is 'static':
            return self._cost_static(circuit)
        elif self.method is 'dynamic':
            return self._cost_dynamic(circuit)

    def t_count_cost(self, circuit):
        """
        Calculate the number of t gates needed if turning a circuit into a Clifford+T circuit.
        Parameterized gates are not supported currently.

        Args:
            circuit(Circuit): Circuit to be measured

        Returns:
            int: the number of t gates needed
        """
        t_count = 0
        for g in circuit.gates:
            if g.type in self.T_COUNT_DICT:
                t_count += self.T_COUNT_DICT[g.type]
            else:
                assert False, f"Gate type: {g.type} not supported by t_count_cost()."
