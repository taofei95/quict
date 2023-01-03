from os import path as osp

from QuICT.core import Circuit, Layout
from QuICT.core.gate import *

from QuICT.qcda.mapping.ai.rl_mapping import RlMapping


def test_main():
    layout_path = osp.join(osp.dirname(__file__), "../layout")
    layout_path = osp.join(layout_path, "grid_3x3.json")
    layout = Layout.load_file(layout_path)
    # It will load model with the same name from model path.
    # If there's no model, you can try train first.
    mapper = RlMapping(layout=layout)
    circ = Circuit(9)
    circ.random_append(random_params=True)
    circ.draw(filename="before_mapping")
    mapped_circ = mapper.execute(circ)
    mapped_circ.draw(filename="mapped_circ")


if __name__ == "__main__":
    test_main()
