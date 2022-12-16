from QuICT.core import Circuit
from QuICT.core.utils import GateType
from QuICT.qcda.optimization.template_optimization import TemplateOptimization
from QuICT.lib.circuitlib import CircuitLib


if __name__ == '__main__':
    circuit = Circuit(5)
    typelist = [
        GateType.x, GateType.cx, GateType.ccx, GateType.h,
        GateType.t, GateType.tdg, GateType.s, GateType.sdg
    ]
    circuit.random_append(50, typelist=typelist)

    circuit.draw(filename='0.jpg')

    while True:
        TO = TemplateOptimization(
            template_max_depth=circuit.width(),
            template_typelist=typelist
        )
        circuit_opt = TO.execute(circuit)
        if circuit_opt.size() == circuit.size():
            break
        circuit = circuit_opt
    circuit_opt.draw(filename='1.jpg')
