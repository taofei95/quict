import importlib


DEVICE_LIST = ["CPU", "GPU"]
BASED_LINALG = ["dot", "tensor", "MatrixTensorI", "MatrixPermutation", "VectorPermutation", "matrix_dot_vector"]
GATE_KERNEL_FUNCTIONS = ["HGate_matrixdot", "CRzGate_matrixdot"]
PROXY_GATE_FUNCTIONS = ["CRzGate_matrixdot_pb", "CRzGate_matrixdot_pc", "CRzGate_matrixdot_pt"]


class LinAlgLoader:
    """
    The Algorithm class with used to load all required algorithm, including based linear algorithm, gate-based matrix dot 
    vector algorithm, and algorithms for multi-GPUs.

    Args:
        device(str): one of ["GPU", "CPU"].
        extra_gate(bool): loading gate-based matrix dot vector algorithm if True.
        extra_proxy(bool): loading the required algorithms of multi-GPUs if True.
    """
    def __init__(self, device: str, extra_gate: bool = False, extra_proxy: bool = False):
        if device not in DEVICE_LIST:
            raise KeyError(f"Not supported the given device, please choice one of {DEVICE_LIST}")

        if device == "CPU":
            linalg_lib = importlib.import_module('QuICT.ops.linalg.cpu_calculator')
        else:
            linalg_lib = importlib.import_module('QuICT.ops.linalg.gpu_calculator')

        for attr, value in linalg_lib.__dict__.items():
            if attr in BASED_LINALG:
                self.__dict__[attr] = value

        if extra_gate:
            gate_lib = importlib.import_module('QuICT.ops.gate_kernel.gate_func')

            for attr, value in gate_lib.__dict__.items():
                if attr in GATE_KERNEL_FUNCTIONS:
                    self.__dict__[attr] = value

        if extra_proxy:
            proxy_lib = importlib.import_module('QuICT.ops.gate_kernel.proxy_gate_func')

            for attr, value in proxy_lib.__dict__.items():
                if attr in PROXY_GATE_FUNCTIONS:
                    self.__dict__[attr] = value
