from typing import List
from numpy import binary_repr

from QuICT.core import Circuit
from QuICT.core.gate import X


def circuit_init(qcircuit: Circuit, qreg_list: List[int], init_val: int):
    """
        Init circuit with a classically given vale using basic/computational encoding in big edian convention.

        Args:
            qcircuit (Circuit): circuit to be inited

            qreg_list (List[int]): the list representing the quantum register to be inited,
            from the most to the least siginificant bit.

            init_val (int): the value used for initialization
    """
    reg_size = len(qreg_list)
    try:
        init_bin = binary_repr(init_val, width=reg_size)
    except Exception as e:
        raise ValueError(f"Fail to convert to binary due to {e}")

    for i, bit in enumerate(init_bin):
        if '1' == bit:
            X | qcircuit([qreg_list[i]])
