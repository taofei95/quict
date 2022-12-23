from os import path as osp

from QuICT.core import *
from QuICT.core.gate import *

from QuICT.qcda.mapping.ai.rl_mapping import RlMapping


def test_main():
    layout_path = osp.join("data", "topo")
    layout_path = osp.join(layout_path, "grid_3x3.json")
    layout = Layout.load_file(layout_path)
    mapper = RlMapping(layout=layout)
    circ = Circuit(9)
    circ.random_append( random_params=True)
    circ.draw(filename="before_mapping")
    mapped_circ = mapper.execute(circ)
    mapped_circ.draw(filename="mapped_circ")


if __name__ == "__main__":
    test_main()
