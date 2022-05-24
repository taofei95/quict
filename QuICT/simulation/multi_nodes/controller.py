from enum import Enum
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from cupy.cuda import nccl

from QuICT.utility import Proxy
from QuICT.core import Circuit
from .transpile import Transpile
from .multi_nodes_simulator import MultiNodesSimulator


class DeviceType(Enum):
    cpu = "CPU",
    gpu = "GPU"


class ModeType(Enum):
    local = "local"
    distributed = "distributed"


class MultiNodesController:
    # Not support noise circuit
    def __init__(
        self,
        ndev: int,
        dev_type: DeviceType,
        mode: ModeType,
        **options
    ):
        self.ndev = ndev
        self._device_type = dev_type
        self._mode_type = mode
        self._transpiler = Transpile(self.ndev)

    def run(self, circuit: Circuit):
        # transpile circuit
        divided_circuits = self._transpiler.run(circuit)

        # start
        if self._mode_type == ModeType.local:
            pass

    def _launch_local(self):
        # Using multiprocess to start simulators, only for GPUs
        pass

    def _launch_distributed(self):
        # send job to distributed
        pass
