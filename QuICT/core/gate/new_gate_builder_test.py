from QuICT.core import *
from QuICT.algorithm import Amplitude
import numpy as np
import random

def _getRandomList(count, upper_bound):
    import random
    _rand = [i for i in range(upper_bound)]
    for i in range(upper_bound - 1, 0, -1):
        do_get = random.randint(0, i)
        _rand[do_get], _rand[i] = _rand[i], _rand[do_get]
    return _rand[:count]

def test_single(n_qubit=10):
    """test for get_gate, get_n_args in gate_builder.py

    Args:
        n_qubit (int, optional): [description]. Defaults to 10.
    """
    type_list = [GATE_ID["Rx"], GATE_ID["Ry"], GATE_ID["Rz"],
                GATE_ID["CX"], GATE_ID["CY"], GATE_ID["CRz"], GATE_ID["CH"], GATE_ID["CZ"],
                GATE_ID["X"],GATE_ID["Y"],GATE_ID["Z"],GATE_ID["ID"],
                GATE_ID["Rxx"], GATE_ID["Ryy"], GATE_ID["Rzz"], GATE_ID["FSim"]
                ]
    gate_list = [Rx, Ry, Rz,
                CX, CY, CRz, CH, CZ,
                X,Y,Z,ID,
                Rxx, Ryy, Rzz, FSim
                ]
    assert len(type_list)==len(gate_list)
    for idx in range(len(type_list)):
        gate_type = type_list[idx]
        circuit1 = Circuit(n_qubit)
        circuit2 = Circuit(n_qubit)

        n_pargs, n_targs, n_cargs = get_n_args(gate_type)
        affect_args = _getRandomList(n_targs+n_cargs, n_qubit)
        pargs = []
        for _ in range(n_pargs):
            pargs.append(random.uniform(0, 2 * np.pi))
        if n_pargs == 0:
            pargs = None

        get_gate(gate_type, affect_args, pargs) | [circuit1[i] for i in affect_args]
        gate_list[idx](pargs) | [circuit2[i] for i in affect_args]

        res1 = Amplitude.run(circuit1)
        res2 = Amplitude.run(circuit2)
        assert np.allclose(res1, res2)

def test_all_in(n_qubit=10):
    type_list = [GATE_ID["Rx"], GATE_ID["Ry"], GATE_ID["Rz"],
                GATE_ID["CX"], GATE_ID["CY"], GATE_ID["CRz"], GATE_ID["CH"], GATE_ID["CZ"],
                GATE_ID["X"],GATE_ID["Y"],GATE_ID["Z"],GATE_ID["ID"],
                GATE_ID["Rxx"], GATE_ID["Ryy"], GATE_ID["Rzz"], GATE_ID["FSim"]
                ]
    gate_list = [Rx, Ry, Rz,
                CX, CY, CRz, CH, CZ,
                X,Y,Z,ID,
                Rxx, Ryy, Rzz, FSim
                ]
    assert len(type_list)==len(gate_list)

    circuit1 = Circuit(n_qubit)
    circuit2 = Circuit(n_qubit)
    for idx in range(len(type_list)):
        gate_type = type_list[idx]

        n_pargs, n_targs, n_cargs = get_n_args(gate_type)
        affect_args = _getRandomList(n_targs+n_cargs, n_qubit)
        pargs = []
        for _ in range(n_pargs):
            pargs.append(random.uniform(0, 2 * np.pi))
        if n_pargs == 0:
            pargs = None

        get_gate(gate_type, affect_args, pargs) | [circuit1[i] for i in affect_args]
        gate_list[idx](pargs) | [circuit2[i] for i in affect_args]

    res1 = Amplitude.run(circuit1)
    res2 = Amplitude.run(circuit2)
    assert np.allclose(res1, res2)

test_single()
test_all_in()