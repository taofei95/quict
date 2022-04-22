from QuICT.core import Circuit
from QuICT.core.utils import GateType
from QuICT.qcda.optimization.template_optimization import TemplateOptimization
from QuICT.qcda.optimization.template_optimization.template_searching.lib import TemplateLib
import time

if __name__ == '__main__':

    t1 = time.perf_counter()
    print(t1)
    circuit = Circuit(3)
    circuit.random_append(rand_size=200, typelist=[GateType.cx, GateType.s, GateType.h])
    circuit.draw(filename='0.jpg')
    print(circuit.size())

    template_lib = TemplateLib(3, 6, 6)
    All_templates = template_lib.template_list()
    while True:
        circuit_opt = TemplateOptimization.execute(circuit, All_templates)
        if circuit_opt.size() == circuit.size():
            break
        circuit = circuit_opt
    circuit_opt.draw(filename='1.jpg')
    t2 = time.perf_counter()
    print(t2, t2-t1)
    print(circuit_opt.size())
