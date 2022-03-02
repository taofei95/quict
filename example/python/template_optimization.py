from QuICT.core import Circuit
from QuICT.core.gate import GateType
from QuICT.qcda.optimization.template_optimization.templates import (template_nct_2a_1,
                                                                     template_nct_2a_2,
                                                                     template_nct_2a_3)
from QuICT.qcda.optimization.template_optimization import TemplateOptimization

if __name__ == '__main__':
    circuit = Circuit(3)
    circuit.random_append(100, typelist=[GateType.x, GateType.cx, GateType.ccx])
    circuit.draw(filename='0.jpg')

    templates = [template_nct_2a_1(), template_nct_2a_2(), template_nct_2a_3()]
    while True:
        circuit_opt = TemplateOptimization.execute(circuit, templates)
        if circuit_opt.size() == circuit.size():
            break
        circuit = circuit_opt
    circuit_opt.draw(filename='1.jpg')
