# 量子门 (Quantum Gate)

量子门是设计用来操作量子比特的状态，基于量子门的线性性质，量子门通常表示为矩阵形式.

!!! example

    例如：考虑一个经典的单比特逻辑门 - 非门，非门的操作是将 $0$ 态和 $1$ 态交换。根据非门的定义，我们可以来定义一个量子非门，即把状态

    $$
    \alpha |0⟩ + \beta |1⟩
    $$

    变换到 $|0⟩$ 和 $|1⟩$ 互换角色的新状态
    
    $$
    \alpha |1⟩ + \beta |0⟩
    $$

    量子非门可以很方便地用矩阵表示，定义矩阵 X 来表示非门如下:

    $$
    X = \begin{bmatrix}
    0&1\\
    1&0\\
    \end{bmatrix}
    $$

    再将量子态 $\alpha |0⟩ + \beta |1⟩$ 写成向量模式为
    
    $$
    \begin{bmatrix}
    \alpha \\
    \beta \\
    \end{bmatrix}
    $$

    其中上面一项对应 |0⟩ 的振幅，下面一项对应 |1⟩ 的振幅，故量子非门的输出为

    $$
    X\begin{bmatrix}
    \alpha \\
    \beta \\
    \end{bmatrix} = \begin{bmatrix}
    \beta \\
    \alpha \\
    \end{bmatrix}
    $$

## 基础量子门 (Basic Gate)

在QuICT中，我们使用 BasicGate 类来实现量子门，包括单/多量子比特门和参数/非参数门。
对于 QuICT 中的每个量子门，它都具有以下属性(以 CX 门为例)：

``` python
from QuICT.core.gate import CX

my_CXGate = CX & [0, 1]     # 构建 CX 量子门，控制位为0，目标位为1
```

### 量子门的基础属性

``` python
my_CXGate.type          # 门的种类
my_CXGate.matrix_type   # 量子门的矩阵类别
my_CXGate.precision     # 门的精度
my_CXGate.matrix        # 量子门矩阵
my_CXGate.target_matrix # 量子门作用位矩阵
my_CXGate.qasm()        # 量子门的 OpenQASM 字符串
```

### 量子门的比特位和参数信息

``` python
my_CXGate.controls     # 控制比特个数
my_CXGate.cargs        # 所有控制比特位
my_CXGate.carg         # 首位控制比特位

my_CXGate.targets      # 目标比特个数
my_CXGate.targs        # 所有目标比特位
my_CXGate.targ         # 首位目标比特位

my_CXGate.params       # 量子门参数个数
my_CXGate.pargs        # 量子门的所有参数
my_CXGate.parg         # 参数首位
```

### 其他量子门操作

``` python
# 部分量子门属性判断
my_CXGate.is_clifford()         # 判断量子门是否属于Clifford
my_CXGate.is_identity()         # 判断量子门矩阵是否为单位矩阵
my_CXGate.is_special()          # 判断门是否为特殊门，如Measure, Reset, Barrier, Unitary, ...

# 量子门交换对比
other_gate = CX & [3, 4]
my_CXGate.commutative(other_gate)

# 量子门变换
inverse_gate = my_CXGate.inverse()          # 得到当前量子门的反转量子门
decomposition_gate = my_CXGate.build_gate() # 将当前量子门分解为若干基础量子门
expand_gate = my_CXGate.expand(qubits=5)    # 将当前量子门扩张为5量子比特大小，即 32 * 32

# QASM
my_CXGate.qasm()    # 得到当前量子门的 QASM 字符串 
```

### 当前 QuICT 所支持的量子门种类

| 量子门种类    | 名称                                                                                        |
| ------------- | ------------------------------------------------------------------------------------------- |
| 单比特门      | H, HY, S, S_dagger, X, Y, Z, SX, SY, SW, ID, U1, U2, U3, Rx, Ry, Rz, Ri, T, T_dagger, Phase |
| 双比特门      | FSim, Rxx, Ryy, Rzz, Rzx, Swap                                                              |
| 受控双比特门  | CZ, CX, CY, CH, CRz, CU1, CU3                                                               |
| 多比特门(>=3) | CCX, CCZ, CCRz, QFT, IQFT, CSwap                                                            |
| 特殊量子门    | Measure, Reset, Barrier, Perm, Unitary, Multi-Control Toffoli, uniformly control gate       |


## 组合量子门 (Composite Gate)

组合量子门, 顾名思义，是多个基础量子门的组合。不同于单一量子门，组合量子门可以实现更加复杂的功能，也通过构建组合量子门这种方式，进一步降低构建量子电路时的复杂性和增加代码的复用性。

### 构建组合量子门

QuICT 通过构建 CompositeGate 类，来实现组合量子门的功能，它存储了量子门的列表。它使用 $or$ ( | ) 和 $xor$ ( ^ ) 追加量子门或逆门，并使用 $and$ (&) 和 $call$ 重新映射控制量子位。同时也支持使用上下文结构来构建量子组合门。

``` python
from QuICT.core.gate import *

# set gate's attributes
cx_gate = CX & [1, 3]   # create CX gate with control qubit 1 and target qubit 3
u2_gate = U2(1, 0)      # create U2 gate with parameters 1 and 0

# create composite gate
cg1 = CompositeGate()
# using default quantum gates
H | cg1(1)                                # append H gate with qubit 1
cx_gate | cg1                             # append pre-defined gates
u2_gate | cg1(0)
U1(1) | cg1(4)                            # append U1 gate with parameters 1 and qubit 4   

# using context to build composite gate
with CompositeGate() as cg_context:
    H & 1
    CX & [1, 3]
    U1(1) & 4

# Add a CompositeGate into CompositeGate
cg_context | cg1

# Draw the CompositeGate into Command Line Interface
cg1.draw('command', flatten=True)
```
``` python
        ┌─────────┐                    
q_0: |0>┤ u2(1,0) ├────────────────────
        └──┬───┬──┘         ┌───┐      
q_1: |0>───┤ h ├───────■────┤ h ├──■───
           └───┘       │    └───┘  │   
q_2: |0>───────────────┼───────────┼───
                     ┌─┴──┐      ┌─┴──┐
q_3: |0>─────────────┤ cx ├──────┤ cx ├
         ┌───────┐ ┌─┴────┴┐     └────┘
q_4: |0>─┤ u1(1) ├─┤ u1(1) ├───────────
         └───────┘ └───────┘           
```
``` python
# Modify the BasicGate in CompositeGate
lgate = cg1.pop()       # Pop the last BasicGate in current CompositeGate
cg1.adjust(1, [0, 2])   # Re-assigned the qubit indexes [0, 2] for the BasicGate with gate index 1 in CompositeGate.
```

在 QuICT 中，我们内置了一些常见的组合量子门，例如： QFT 、多控 Toffoli 门、 CCRz 等量子门组合。

``` python
from QuICT.core.gate import QFT, MultiControlToffoli

qft_gate = QFT(3)               # 构建 3 比特 QFT 量子门
qft_gate.draw('command')        # 画出 3 比特 QFT 量子门

mct = MultiControlToffoli('no_aux')
mct_gate = mct(3)               # 构建 3 比特多控 toffoli 门，不使用辅助比特
```
``` python
        ┌───┐┌─────┐┌─────┐                 
q_0: |0>┤ h ├┤ cu1 ├┤ cu1 ├─────────────────
        └───┘└──┬──┘└──┬──┘┌───┐┌─────┐     
q_1: |0>────────■──────┼───┤ h ├┤ cu1 ├─────
                       │   └───┘└──┬──┘┌───┐
q_2: |0>───────────────■───────────■───┤ h ├
                                       └───┘
```

### 量子组合门常用属性

在某种程度上，量子组合门可以被视作一个没有量子比特信息的量子电路。所以与量子电路类似，量子组合门也包含与量子电路类似的基础属性。

``` python
cg1.size()      # The number of Quantum Gates in current CompositeGate
cg1.depth()     # The depth of current CompositeGate
cg1.width()     # The number of qubits in current CompositeGate
cg1.qubits()    # The qubits indexes of current CompositeGate
```
``` python
7 4 4 [0, 1, 3, 4]
```
```python
inv_cg1 = cg1.inverse()             # Get the inverse of CompositeGate
cg1_matrix = cg1.matrix(local=True) # Get the matrix of current CompositeGate with local qubit indexes
```
``` python
# The inverse CompositeGate
        ┌──────────────┐                     
q_0: |0>┤ u2(π,2.1416) ├─────────────────────
        └──────────────┘  ┌───┐         ┌───┐
q_1: |0>───────■──────────┤ h ├─────■───┤ h ├
               │          └───┘     │   └───┘
q_2: |0>───────┼────────────────────┼────────
             ┌─┴──┐               ┌─┴──┐     
q_3: |0>─────┤ cx ├───────────────┤ cx ├─────
           ┌─┴────┴─┐   ┌────────┐└────┘     
q_4: |0>───┤ u1(-1) ├───┤ u1(-1) ├───────────
           └────────┘   └────────┘           
```


## 量子指令集 （Instruction Set）
对于每一个量子机来说，都会有一组由若干单比特量子门和一个双比特量子门组成的指令集，用来控制和操作量子态，从而实现所需要的量子计算。量子指令集的设计和实现对于量子计算机来说是非常重要的，它关系到量子计算应用的速度、精度、效果和可重复性。

在 QuICT 中，我们构建了一个 InstructionSet 类用来存储一组量子指令集，除了对自定义指令集的支持以外，针对当前行业内常见的量子指令集，QuICT已内置了相应的实现，例如USTCSet、GoogleSet、IBMQSet等。另一方面，QuICT也实现了诸多量子门转换算法，支持将任意量子门电路转换为当前指令集电路，即只含有指令集内量子门的电路。这部分在QCDA中会有更详细的展示。

``` python
from QuICT.core.utils import GateType
from QuICT.core.virtual_machine import InstructionSet
from QuICT.core.virtual_machine.special_set import USTCSet


single_qubit_gates = [GateType.h, GateType.rx, GateType.ry, GateType.rz]
double_qubit_gate = GateType.cx

iset = InstructionSet(
    two_qubit_gate=double_qubit_gate,
    one_qubit_gates=single_qubit_gates,
    one_qubit_rule=None
)
print(iset.gates)
print(USTCSet.gates)
```
``` python
[<GateType.h: 'H gate'>, <GateType.rx: 'Rx gate'>, <GateType.ry: 'Ry gate'>, <GateType.rz: 'Rz gate'>, <GateType.cx: 'controlled-X gate'>]
[<GateType.rx: 'Rx gate'>, <GateType.ry: 'Ry gate'>, <GateType.rz: 'Rz gate'>, <GateType.h: 'H gate'>, <GateType.x: 'Pauli-X gate'>, <GateType.cx: 'controlled-X gate'>]
```
