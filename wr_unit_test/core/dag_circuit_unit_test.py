# import numpy as np
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.utils import GateType

def main():
        circuit = Circuit(5)
        circuit.random_append(20)
        # circuit.draw()
        print(circuit.qasm())
        dag_cir = circuit.get_DAG_circuit()
        # dag_cir.draw()

        dag_cir.size == circuit.size()
        # dag_cir.width == circuit.width()



        edge_list = dag_cir.edges()
        print(edge_list)

        gs = circuit.gates
        for start, end in edge_list:
            assert not gs[start].commutative(gs[end])
            # forward/backward
            forward = True
        for f in range(start + 1, end, 1):
            if not gs[start].commutative(gs[f]):
                forward = False
                break

        if not forward:
            for b in range(end - 1, start, -1):
                if not gs[end].commutative(gs[b]):
                    return False
        return True



if __name__ == '__main__':
   print(main())
   # a = Ryy & [3,2]
   # b = Ry & 2
   # print(a.commutative(b))