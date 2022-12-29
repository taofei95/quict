import random

from QuICT.core import Circuit
from QuICT.core.utils import GateType
from QuICT.qcda.optimization.template_optimization import TemplateOptimization


if __name__ == '__main__':
    circuit = Circuit(5)

    typelist = [GateType.x, GateType.cx, GateType.ccx,
                GateType.h, GateType.s, GateType.t, GateType.sdg, GateType.tdg]
    circuit.random_append(200, typelist=typelist)
    circuit.draw(filename='0.jpg')

    TO = TemplateOptimization()
    circ_optim = TO.execute(circuit)
    circ_optim.draw(filename='1.jpg')
