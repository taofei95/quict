import numpy as np
from typing import List
import inspect

from QuICT.core import *
from QuICT.qcda.optimization._optimization import Optimization
from dag import DAG
from phase_poly import PhasePolynomial
from template import *


class AutoOptimization(Optimization):
    """
    Heuristic optimization of circuits in Clifford + Rz.

    [1] Nam, Yunseong, et al. "Automated optimization of large quantum
    circuits with continuous parameters." npj Quantum Information 4.1
    (2018): 1-12.
    """

    _optimize_sub_method = {
        1: "reduce_hadamard_gates",
        2: "cancel_single_qubit_gates",
        3: "cancel_two_qubit_gates",
        4: "merge_rotations",
        5: "float_rotations",
    }
    _optimize_routine = {
        'heavy': [1, 3, 2, 3, 1, 2, 5],
        'light': [1, 3, 2, 3, 1, 2, 4, 3, 2],
    }

    @classmethod
    def reduce_hadamard_gates(cls, gates: DAG):
        cnt = 0
        # enumerate templates and replace every occurrence
        for template in hadamard_templates:
            cnt += template.replace_all(gates)
        return cnt

    @classmethod
    def cancel_single_qubit_gates(cls, gates: DAG, epsilon=1e-8):
        cnt = 0
        for node in gates.topological_sort():
            # enumerate every single qubit gate
            if node.gate.qasm_name != 'rz':
                continue
            # erase the gate if degree = 0
            if abs(node.gate.parg) < epsilon:
                node.erase()
                cnt += 1
                continue

            # try cancelling while commuting the gate with templates
            # (c_node, c_qubit): the position right before template matching
            c_node, c_qubit = node, 0
            while True:
                # (n_node, n_qubit): the start of template matching
                n_node, n_qubit = c_node.successors[c_qubit]
                # stop if reaching the end of circuit
                if not n_node.gate:
                    break

                # if n_node is another rz, merge and erase the original node
                if n_node.gate.qasm_name == node.gate.qasm_name:
                    n_node.gate.parg += node.gate.parg
                    node.erase()
                    cnt += 1
                    break

                # template matching
                mapping = None
                for template in single_qubit_gate_templates:
                    mapping = mapping or template.compare(c_node.successors[c_qubit])
                    if mapping:
                        # found a sub-circuit that commutes
                        # set (c_node, c_qubit) to be the last position of this sub-circuit
                        c_node, c_qubit = template.template.end_nodes[template.anchor].predecessors[0]
                        c_node = mapping[id(c_node)]
                        break
                # found no templates. commuting fails
                if not mapping:
                    break
        return cnt

    @classmethod
    def cancel_two_qubit_gates(cls, gates: DAG):
        cnt = 0
        for node in list(gates.topological_sort()):
            if node.flag == DAG.Node.FLAG_ERASED or node.gate.qasm_name != 'cx':
                continue

            c_ctrl_node, c_ctrl_qubit = node, 0
            c_targ_node, c_targ_qubit = node, 1
            while True:
                n_ctrl_node, n_ctrl_qubit = c_ctrl_node.successors[c_ctrl_qubit]
                n_targ_node, n_targ_qubit = c_targ_node.successors[c_targ_qubit]
                if id(n_ctrl_node) == id(n_targ_node) and n_ctrl_node.gate.qasm_name == 'cx' and \
                        n_ctrl_qubit == 0 and n_targ_qubit == 1:
                    n_ctrl_node.erase()
                    node.erase()
                    cnt += 2

                mapping = None
                for template in cnot_ctrl_template:
                    mapping = mapping or template.compare(c_ctrl_node.successors[c_ctrl_qubit])
                    if mapping:
                        c_ctrl_node, c_ctrl_qubit = template.template.end_nodes[template.anchor].predecessors[0]
                        c_ctrl_node = mapping[id(c_ctrl_node)]
                        break
                if mapping:
                    continue
                for template in cnot_targ_template:
                    mapping = mapping or template.compare(c_targ_node.successors[c_targ_qubit])
                    if mapping:
                        c_targ_node, c_targ_qubit = template.template.end_nodes[template.anchor].predecessors[0]
                        c_targ_node = mapping[id(c_targ_node)]
                        break
                if not mapping:
                    break
        return cnt

    @classmethod
    def merge_rotations(cls, gates: DAG):
        # TODO S, Sdg, T, Tdg can be included
        gate_set = {'rz', 'cx', 'x'}
        for prev_node, succ_node in list(gates.enumerate_sub_circuit(gate_set)):
            phase_poly = PhasePolynomial(DAG.create_sub_circuit(prev_node, succ_node))
            replacement = DAG(phase_poly.get_circuit())

            mapping = {}
            for qubit_ in gates.size:
                if not prev_node[qubit_] or not succ_node[qubit_]:
                    continue
                mapping[id(replacement.start_nodes[qubit_])] = prev_node[qubit_]
                mapping[id(replacement.end_nodes[qubit_])] = succ_node[qubit_]

            DAG.replace_circuit(mapping, replacement)

    @classmethod
    def float_rotations(cls, gates: DAG):
        print(inspect.currentframe(), 'not implemented yet')

    @classmethod
    def _execute(cls, gates, routine: List[int]):
        _gates = DAG(CompositeGate(gates))
        cnt = 0
        while True:
            for step in routine:
                cnt += getattr(cls, cls._optimize_sub_method[step])(_gates)
            if cnt == 0:
                break

        return _gates.get_circuit()

    @classmethod
    def execute(cls, gates, mode='light'):
        """
        Heuristic optimization of circuits in Clifford + Rz.

        Args:
              gates(Union[Circuit, CompositeGate]): Circuit to be optimized
              mode(str): Support 'light' and 'heavy' mode. See details in [1].
        Returns:
            CompositeGate: The CompositeGate after optimization

        [1] Nam, Yunseong, et al. "Automated optimization of large quantum
        circuits with continuous parameters." npj Quantum Information 4.1
        (2018): 1-12.
        """
        if mode in cls._optimize_routine:
            return cls._execute(gates, cls._optimize_routine[mode])
        else:
            raise Exception(f'unrecognized mode {mode}')
