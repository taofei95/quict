import os
import yaml
from enum import Enum

from QuICT.core import GATE_ID


class GateType(Enum):
    matrix_1arg = "2x2Matrix"
    matrix_2arg = "4x4Matrix"
    diagonal_1arg = "2x2Diagonal"
    diagonal_2arg = "4x4Diagonal"
    swap_1arg = "2x2RDiagonal"
    swap_2arg = "4x4Swap"
    swap_3arg = "8x8Swap"
    control_1arg = "2x2Control"
    control_2arg = "4x4Control"
    control_3arg = "8x8Control"
    reverse_1arg = "2x2Reverse"
    reverse_2arg = "4x4Reverse"
    reverse_3arg = "8x8Reverse"
    complexMIP_2arg = "4x4ComplexMIP"
    complexIPIP_2arg = "4x4ComplexIPIP"


GATE_TYPE_to_ID = {
    GateType.matrix_1arg: [
        GATE_ID["H"], GATE_ID["SX"], GATE_ID["SY"], GATE_ID["SW"],
        GATE_ID["U2"], GATE_ID["U3"], GATE_ID["RX"], GATE_ID["RY"]
    ],
    GateType.matrix_2arg: [GATE_ID["CH"], GATE_ID["CU3"]],
    GateType.diagonal_1arg: [GATE_ID["RZ"], GATE_ID["Phase"]],
    GateType.diagonal_2arg: [GATE_ID["RZZ"]],
    GateType.swap_1arg: [GATE_ID["X"]],
    GateType.swap_2arg: [GATE_ID["Swap"]],
    GateType.swap_3arg: [GATE_ID["CSwap"]],
    GateType.reverse_1arg: [GATE_ID["Y"]],
    GateType.reverse_2arg: [GATE_ID["CX"], GATE_ID["CY"]],
    GateType.reverse_3arg: [GATE_ID["CCX"]],
    GateType.control_1arg: [
        GATE_ID["Z"], GATE_ID["U1"], GATE_ID["T"],
        GATE_ID["T_dagger"], GATE_ID["S"], GATE_ID["S_dagger"]
    ],
    GateType.control_2arg: [GATE_ID["CZ"], GATE_ID["CU1"]],
    GateType.control_3arg: [GATE_ID["CCRz"]],
    GateType.complexMIP_2arg: [GATE_ID["FSim"]],
    GateType.complexIPIP_2arg: [GATE_ID["RXX"], GATE_ID["RYY"]]
}


MATRIX_INDEXES = [
    [10, 11, 14, 15],
    [5, 6, 9, 10],
    [10, 9, 6, 5],
    [0, 3, 12, 15],
    [36, 37, 38, 39, 44, 45, 46, 47, 52, 53, 54, 55, 60, 61, 62, 63],
    [45, 46, 62, 63],
    [36, 37, 53, 54],
    [54, 55, 62, 63]
]


def _get_default_config():
    """ Generate the dict of the default config for the simulator.

    Returns:
        [dict]: The default config dict for simulator.
    """
    curPath = os.path.dirname(os.path.realpath(__file__))
    simPath = os.path.split(curPath)[0]
    confPath = os.path.join(simPath, "config", "default.yml")

    with open(confPath, 'r', encoding='utf-8') as f:
        config = f.read()

    config = yaml.load(config, Loader=yaml.FullLoader)

    return config


_DEFAULT_CONFIG = _get_default_config()


def option_validation():
    """ Check options' correctness for specified simulator. """
    def decorator(func):
        def wraps(self, *args, **kwargs):
            device = self._device
            backend = self._backend
            if device == "GPU":
                default_options = _DEFAULT_CONFIG[device][backend]
            else:
                default_options = _DEFAULT_CONFIG[device]

            customized_options = kwargs["options"]
            if customized_options.keys() - default_options.keys():
                raise KeyError(f"There are some unsupportted options in current simulator. {customized_options}")
            else:
                return func(self, customized_options, default_options)

        return wraps

    return decorator
