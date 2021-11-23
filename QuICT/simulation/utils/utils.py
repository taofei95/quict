import subprocess
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


def get_bandwidth():
    default = 10

    try:
        # Get nvidia device ip
        ret = subprocess.check_output(['lspci | grep -i nvidia'], shell=True)
        ret = ret.decode()
        device_ip = ret.split(" ")[0]

        # Get vendor ip
        ret = subprocess.check_output([f'lspci -n | grep -i {device_ip}'], shell=True)
        ret = ret.decode()
        vendor_ip = ret.split(" ")[2]

        # Get bandwidth
        ret = subprocess.check_output([f'lspci -n -d {vendor_ip} -vvv | grep -i width'], shell=True)
        ret = ret.decode()
    except Exception as _:
        return default


def set_buffsize(qubits: int):
    import os

    buffsize = (1 << qubits) * 8    # Get buffsize by the given qubits with complex64
    user_path = os.path.expanduser('~')
    with open(f"{user_path}/.nccl.conf", mode="w") as f:
        f.write(f"NCCL_BUFFSIZE={buffsize}")
