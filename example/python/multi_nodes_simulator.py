from copy import deepcopy
import os
import numpy as np

from QuICT.core.gate import *
from QuICT.simulation.multi_nodes.multi_nodes_simulator import MultiNodesSimulator
from QuICT.simulation.unitary.unitary_simulator import UnitarySimulator
from QuICT.tools.interface import OPENQASMInterface
from QuICT.simulation.multi_nodes.controller import MultiNodesController

cir = OPENQASMInterface.load_file(
    os.path.dirname(os.path.abspath(__file__)) +
    "../../unit_test/simulation/data/random_circuit_for_correction.qasm"
).circuit
sv_data = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)) +
    "../../unit_test/simulation/data/state_vector.npy"
)
sv_data_single = sv_data.astype(np.complex64)

multi_simulator = MultiNodesController(
    ndev=2,
    matrix_aggregation=False,
    precision="double"
)


def test_double():
    sim = MultiNodesController(2, matrix_aggregation=False, precision="double")
    m = sim.run(cir).get()
    assert np.allclose(m, sv_data)


def test_single():
    sim = MultiNodesController(2, matrix_aggregation=False, precision="single")
    sv = sim.run(cir).get()
    assert np.allclose(sv, sv_data_single, atol=1e-7)
