from os import path as osp
from typing import List, Tuple, Union
from QuICT.core import *

from QuICT.core.gate import *
from QuICT.qcda.mapping import MCTSMapping
from QuICT.simulation.state_vector import StateVectorSimulator

CircuitLike = Union[Circuit, CompositeGate]


def _quick_sort(arr: List[int], begin: int, end: int, swaps: List[Tuple[int, int]]):
    """Construct swap sequence using quick sort algorithm.

    Args:
        arr (List[int]): Array to be sorted.
        begin (int): Begin index(included).
        end (int): End index(excluded).
        swaps (List[Tuple[int, int]]): Output argument. Result swaps will be appended into.
    """
    if end <= begin:
        return
    pivot = begin
    for i in range(begin, end):
        if arr[i] < arr[begin]:
            pivot += 1
            if i != pivot:
                arr[i], arr[pivot] = arr[pivot], arr[i]
                swaps.append((i, pivot))
    if begin != pivot:
        arr[begin], arr[pivot] = arr[pivot], arr[begin]
        swaps.append((begin, pivot))
    _quick_sort(arr, begin, pivot, swaps)
    _quick_sort(arr, pivot + 1, end, swaps)


def _wrap_to_circ(circuit_like: CircuitLike, width: int) -> Circuit:
    circ = Circuit(width)
    circuit_like | circ(list(range(width)))
    return circ


def check_circ_eq(lhs: Circuit, rhs: Circuit) -> bool:
    simulator = StateVectorSimulator()
    state_1 = simulator.run(circuit=lhs)
    state_2 = simulator.run(circuit=rhs)
    assert np.allclose(state_1, state_2)


def check_circ_mapped(circ: Circuit, layout: Layout) -> bool:
    allowed_positions = set()
    for pos in layout.directionalized:
        allowed_positions.add((pos.u, pos.v))
    for gate in circ.flatten_gates():
        if gate.controls + gate.targets == 1:
            continue
        pos = tuple(gate.cargs + gate.targs)
        assert pos in allowed_positions


def test_mapping():
    file_dir = osp.dirname(osp.abspath(__file__))
    # layout_names = ["ibmq_casablanca", "lnn20", "ibmq20"]
    layout_names = ["ibmq_casablanca"]
    for layout_name in layout_names:
        layout_path = osp.join(file_dir, "example")
        layout_path = osp.join(layout_path, f"{layout_name}.json")
        layout = Layout.load_file(layout_path)
        for _ in range(3):
            q = layout.qubit_number
            circ = Circuit(q)
            circ.random_append(
                20,
                random_params=True,
            )

            mapper = MCTSMapping(layout=layout)
            mapped_circ = mapper.execute(circ)
            phy2logic = mapper.phy2logic

            check_circ_mapped(mapped_circ, layout)

            swaps = []
            _quick_sort(arr=phy2logic, begin=0, end=q, swaps=swaps)
            remapped_circ = _wrap_to_circ(mapped_circ, q)
            for pos in swaps:
                Swap | remapped_circ(list(pos))

            check_circ_eq(circ, remapped_circ)
