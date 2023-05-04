from QuICT.core.virtual_machine import InstructionSet
from QuICT.core.virtual_machine.special_set import USTCSet
from QuICT.core.utils import GateType


def build_iset():
    single_qubit_gates = [GateType.h, GateType.rx, GateType.ry, GateType.rz]
    double_qubit_gate = GateType.cx

    iset = InstructionSet(
        two_qubit_gate=double_qubit_gate,
        one_qubit_gates=single_qubit_gates,
        one_qubit_rule=None
    )
    print(iset.gates)

    # Set Single-Qubit Gates' Rule, you can define yourself function rule or one of 
    # [zyz_rule, zxz_rule, hrz_rule, xyx_rule, ibmq_rule, u3_rule]
    print(iset.one_qubit_rule)
    iset.register_one_qubit_rule('hrz_rule')
    print(iset.one_qubit_rule)

    # Pre-build Instruction Set
    print(USTCSet.gates)
    print(USTCSet.one_qubit_rule)


if __name__ == "__main__":
    build_iset()
