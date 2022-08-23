from QuICT.core import Circuit
from QuICT.core.utils import GateType
from QuICT.qcda.optimization.template_optimization.templates import (template_nct_2a_2,
                                                                     template_nct_4a_3,
                                                                     template_nct_5a_3,
                                                                     template_nct_6a_1,
                                                                     template_nct_9c_5,
                                                                     template_nct_9d_4)
from QuICT.qcda.optimization.template_optimization import TemplateOptimization


if __name__ == '__main__':
    circuit = Circuit(5)
    circuit.random_append(50, typelist=[GateType.cx])
    circuit.draw(filename='0.jpg')

    templates = [template_nct_2a_2(),
                 template_nct_4a_3(),
                 template_nct_5a_3(),
                 template_nct_6a_1(),
                 template_nct_9c_5(),
                 template_nct_9d_4()]
    while True:
        TO = TemplateOptimization(templates)
        circuit_opt = TO.execute(circuit)
        if circuit_opt.size() == circuit.size():
            break
        circuit = circuit_opt
    circuit_opt.draw(filename='1.jpg')
