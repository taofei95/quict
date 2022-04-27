from enum import Enum

from QuICT.core import Circuit


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

    def _circuit_transpile(self, circuit):
        pass

    def run(self, circuit: Circuit):
        # transpile circuit
        divided_circuits = self._circuit_transpile(circuit)

        # start
        # simulator + proxy [decided by device and mode]
        pass

    def _launch_local(self):
        # Using multiprocess to start simulators, only for GPUs
        pass
    
    def _launch_distributed(self):
        # send job to distributed
        pass
