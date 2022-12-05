import random

from QuICT.core import Circuit
from QuICT.core.utils import GateType
from QuICT.tools.circuit_library import CircuitLib
from QuICT.qcda.optimization.template_optimization import TemplateOptimization


if __name__ == '__main__':
    circuit = Circuit(5)
    typelist = [GateType.x, GateType.cx, GateType.ccx,
                GateType.h, GateType.s, GateType.t, GateType.sdg, GateType.tdg]
    circuit.random_append(200, typelist=typelist)
    circuit.draw(filename='before_temp')

    template_list = CircuitLib().get_template_circuit()
    TO = TemplateOptimization(
        template_list=random.sample(template_list, 10),
        heuristics_qubits_param=[10],
        heuristics_backward_param=[3, 1]
    )
    circ_optim = TO.execute(circuit)
    circ_optim.draw(filename='after_temp')