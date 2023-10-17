import random

from QuICT.core.circuit.circuit import Circuit
from QuICT.core.gate import *

qubit = 5
cir = Circuit(qubit)  # 5比特电路
cgate = CompositeGate(name="1")  # 组合门比特数不限制
SX | cgate(1)
CZ | cgate([1, 2])
CCRz | cgate([1, 2, 3])  # 只有3个比特位上出现门
cgate | cir([0, 1, 2])  # 需要3个index
#         ┌───────┐
# q_0: |0>┤0      ├
#         │       │
# q_1: |0>┤1 cg_1 ├
#         │       │
# q_2: |0>┤2      ├
#         └───────┘
# q_3: |0>─────────

# q_4: |0>─────────

qubit = 5
cir1 = Circuit(qubit)  # 5比特电路
cgate1 = CompositeGate(name="2")  # 组合门比特数不限制
SX | cgate1(1)
CX | cgate1([7, 8])  # 5个比特位上出现门
CCRz | cgate1([1, 2, 3])
cgate1 | cir1([0, 1, 2, 3, 4])  # 需要5个index
#         ┌───────┐
# q_0: |0>┤0      ├
#         │       │
# q_1: |0>┤1      ├
#         │       │
# q_2: |0>┤2 cg_2 ├
#         │       │
# q_3: |0>┤3      ├
#         │       │
# q_4: |0>┤4      ├
#         └───────┘

qubit = 5
cir2 = Circuit(qubit)  # 5比特电路
cgate2 = CompositeGate(name="3")  # 组合门比特数不限制
SX | cgate2(1)
CX | cgate2([2, 3])  # 3个比特位上出现门
cgate2 | cir2([3, 2, 1])  # index不是顺序的，哪里翻转组合门的index也翻转
# q_0: |0>─────────
#         ┌───────┐
# q_1: |0>┤2      ├
#         │       │
# q_2: |0>┤1 cg_3 ├
#         │       │
# q_3: |0>┤0      ├
#         └───────┘
# q_4: |0>─────────

qubit = 5
cir3 = Circuit(qubit)  # 5比特电路
cgate3 = CompositeGate(name="4")  # 组合门比特数不限制
for _ in range(5):  # 多试一下重复插入组合门
    SX | cgate3(1)
    CX | cgate3([1, 2])
    cgate3 | cir3(random.sample(list(range(qubit)), 2))
cir3.draw("command")
#         ┌───────┐                           ┌───────┐
# q_0: |0>┤1      ├───────────────────────────┤1      ├
#         │       │                           │       │
# q_1: |0>┤  cg_4 ├───────────────────────────┤       ├
#         │       │┌───────┐┌───────┐┌───────┐│  cg_4 │
# q_2: |0>┤0      ├┤1      ├┤0      ├┤0      ├┤       ├
#         └───────┘│       ││       ││       ││       │
# q_3: |0>─────────┤  cg_4 ├┤  cg_4 ├┤  cg_4 ├┤0      ├
#                  │       ││       ││       │└───────┘
# q_4: |0>─────────┤0      ├┤1      ├┤1      ├─────────
#                  └───────┘└───────┘└───────┘   

#  测的时候改变一下电路种类测