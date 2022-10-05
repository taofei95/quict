from QuICT.core import Circuit
from .template_substitution import TemplateSubstitution
from .template_matching import TemplateMatching
from .template_matching import MatchingDAGCircuit, Match
from QuICT.qcda.optimization._optimization import Optimization


class TemplateOptimization(Optimization):
    """
    Class for the template optimization pass.
    """
    @classmethod
    def execute(
        cls,
        circuit,
        template_list,
        heuristics_qubits_param=None,
        heuristics_backward_param=None
    ):

        circ_dag = MatchingDAGCircuit(circuit)

        for template in template_list:
            template_dag = MatchingDAGCircuit(template)

            if template_dag.width > circ_dag.width:
                continue

            matches = TemplateMatching.execute(
                circ_dag,
                template_dag,
                heuristics_qubits_param,
                heuristics_backward_param
            )

            circ_dag = TemplateSubstitution.execute(circ_dag, template_dag, matches)

        return circ_dag.get_circuit()
