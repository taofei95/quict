import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from cupy.cuda import nccl
import numpy as np

from QuICT.utility import Proxy
from .multi_nodes_simulator import MultiNodesSimulator


if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)


def worker(ndev, uid, dev_id, circuit):
    proxy = Proxy(ndevs=ndev, uid=uid, dev_id=dev_id)
    simulator = MultiNodesSimulator(
        proxy=proxy,
        device="GPU",
        gpu_id=dev_id
    )
    state = simulator.run(circuit)

    return dev_id, state.get()


class MultiNodeLauncher:
    def __init__(self, ndev: int, precision: str = "double", sync: bool = True):
        self.ndev = ndev
        self.precision = precision
        self.sync = sync

    def run(self, circuit, *args):
        uid = nccl.get_unique_id()
        with ProcessPoolExecutor(max_workers=self.ndev) as executor:
            tasks = [
                executor.submit(
                    worker,
                    self.ndev,
                    uid,
                    dev_id,
                    circuit
                ) for dev_id in range(self.ndev)
            ]

        results = []
        for t in as_completed(tasks):
            results.append(t.result())

        z = [None] * self.ndev
        for idx, vec in results:
            z[idx] = vec

        return np.concatenate(z)
