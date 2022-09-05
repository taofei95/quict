import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.tools.interface import OPENQASMInterface
from QuICT.simulation.multi_nodes.controller import MultiNodesController


circuit = OPENQASMInterface.load_file("./temp_test_script/cswap_dcom.qasm").circuit
multi_simulator = MultiNodesController(
    ndev=2,
    matrix_aggregation=False,
    precision="double"
)

sv = multi_simulator.run(circuit)
