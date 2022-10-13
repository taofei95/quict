from QuICT.qcda.optimization.template_optimization.templates import \
    template_nct_2a_1, template_nct_2a_2, template_nct_2a_3

from QuICT.core import Circuit
from .template_substitution import TemplateSubstitution
from .template_matching import TemplateMatching
from .template_matching import MatchingDAGCircuit
from QuICT.qcda.optimization._optimization import Optimization


class TemplateOptimization(Optimization):

    @classmethod
    def execute(
        cls,
        circuit,
        template_list=None,
        heuristics_qubits_param=None,
        heuristics_backward_param=None
    ):

        if template_list is None:
            template_list = [template_nct_2a_1(), template_nct_2a_2(), template_nct_2a_3()]

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
