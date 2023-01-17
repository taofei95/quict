import os
from QuICT.core.circuit.circuit import Circuit
from QuICT.core.layout.layout import Layout
from QuICT.core.utils.gate_type import GateType
from QuICT.qcda.mapping import (
    MCTSMapping,
    SABREMapping)
# from QuICT.qcda.mapping.ai.rl_mapping import RlMapping

def gate_count(circuit):
    size = circuit.count_gate_by_gatetype(GateType.swap)
    return size

layout_grid = Layout.load_file(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "../example/layout/ibmqx2_layout.json"))
layout_ibmqx2 = Layout.load_file(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "../example/layout/ibmqx2_layout.json"))
layout_ibmq_lima = Layout.load_file(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "../example/layout/ibmq_lima.json"))
layout_file = [layout_grid, layout_ibmqx2, layout_ibmq_lima]
layout_file_name = ["layout_grid", "layout_ibmqx2", "layout_ibmq_lima"]
gates = [5, 10, 15, 20]

if __name__ == '__main__':
    data = open("mapping_benchmark_data.txt", 'w+')
    
    for g in gates:
        circuit = Circuit(5)
        circuit.random_append(5 * g, typelist=[
            GateType.cx, GateType.cz, GateType.ch, GateType.crz, GateType.rzx,
            GateType.rxx, GateType.ryy, GateType.rzz, GateType.cu1, GateType.cu3
        ])
        data.write(f"gate number:{g} \n")
        for f in layout_file:
            data.write(f"{layout_file_name[layout_file.index(f)]} \n")
            # mcts
            mcts = MCTSMapping(f)
            circuit_map = mcts.execute(circuit)
            data.write(f"mcts: {gate_count(circuit_map)} \n")

            # sabre
            sabre = SABREMapping(f)
            circuit_map = sabre.execute(circuit)
            data.write(f"sabre: {gate_count(circuit_map)} \n")

            # # rl
            # mapper = RlMapping(layout=f)
            # mapped_circ = mapper.execute(circuit)
            # data.write(f"rl:", gate_count(circuit_map))

    data.close()