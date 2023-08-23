from typing import List, Union
from numpy import binary_repr

from QuICT.core import Circuit
from QuICT.core.gate import BasicGate, CompositeGate
from QuICT.core.gate import X


def circuit_init(circuit: Circuit, qreg_list: List[int], init_val: int):
    """
        Init circuit with a classically given vale using basic/computational encoding in big edian convention.

        Args:
            circuit (Circuit): circuit to be initialized.

            qreg_list (List[int]): the list representing the quantum register to be initialized,
            from the most to the least siginificant bit.

            init_val (int): the integer value used for initialization
    """
    reg_size = len(qreg_list)
    try:
        init_bin = binary_repr(init_val, width=reg_size)
    except Exception as e:
        raise ValueError(f"Fail to convert to binary due to {e}")

    for i, bit in enumerate(init_bin):
        if '1' == bit:
            X | circuit([qreg_list[i]])


def decompose(circuit: Circuit, level: int = -1) -> Circuit:
    """
        Decompose composite gate in a circuit.

        Args:
            circuit (Circuit): circuit to be decomposed.

            level (int): decomposition level. Negative number for complete
            decomposition. Default to be -1.

        Returns:
            Circuit: The decomposed circuit.
    """
    new_circ = Circuit(circuit.width())

    for gate, qidxes, _ in decompose_gate_list(circuit._gates, level):
        gate | new_circ(qidxes)

    return new_circ


def decompose_gate_list(
    gate_list: List[Union[BasicGate, CompositeGate]],
    level: int = -1
) -> List[Union[BasicGate, CompositeGate]]:
    """
        Decompose composite gate in a list containing basic gates and composite gates.

        Args:
            gate_list (List[Union[BasicGate, CompositeGate]]): the list of gate to be
            decomposed.

            level (int): decomposition level. Negative number for complete
            decomposition. Default to be -1.

        Returns:
            List[Union[BasicGate, CompositeGate]]: The decomposed gate list.
    """
    decomp_gates = []
    for gate, qidxes, size in gate_list:
        if size > 1 or hasattr(gate, "gate_decomposition"):
            if level < 0:
                temp_gate = gate.copy() & qidxes
                decomp_gates += decompose_gate_list(temp_gate._gates)
            elif level == 0:
                decomp_gates.append((gate, qidxes, size))
            else:
                temp_gate = gate.copy() & qidxes
                decomp_gates += decompose_gate_list(temp_gate._gates, level - 1)
        else:
            decomp_gates.append((gate, qidxes, size))

    return decomp_gates
