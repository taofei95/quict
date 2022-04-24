import os
import yaml
from enum import Enum

from QuICT.core.utils import GateType


class GateGroup(Enum):
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
    perm_gate = "PermGateClass"


GATE_TYPE_to_ID = {
    GateGroup.matrix_1arg: [
        GateType.h, GateType.sx, GateType.sy, GateType.sw,
        GateType.u2, GateType.u3, GateType.rx, GateType.ry
    ],
    GateGroup.matrix_2arg: [GateType.ch, GateType.cu3],
    GateGroup.diagonal_1arg: [GateType.rz, GateType.phase],
    GateGroup.diagonal_2arg: [GateType.Rzz],
    GateGroup.swap_1arg: [GateType.x],
    GateGroup.swap_2arg: [GateType.swap],
    GateGroup.swap_3arg: [GateType.cswap],
    GateGroup.reverse_1arg: [GateType.y],
    GateGroup.reverse_2arg: [GateType.cx, GateType.cy],
    GateGroup.reverse_3arg: [GateType.ccx],
    GateGroup.control_1arg: [
        GateType.z, GateType.u1, GateType.t,
        GateType.tdg, GateType.s, GateType.sdg
    ],
    GateGroup.control_2arg: [GateType.cz, GateType.cu1],
    GateGroup.control_3arg: [GateType.CCRz],
    GateGroup.complexMIP_2arg: [GateType.fsim],
    GateGroup.complexIPIP_2arg: [GateType.Rxx, GateType.Ryy]
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
    """ Check options' correctness for the given simulator. """
    def decorator(func):
        def wraps(self, **kwargs):
            device = self._device
            backend = self._backend
            if device in ["GPU", "CPU"]:
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
