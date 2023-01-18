# 量子门 (Quantum Gate)
量子门是设计用来操作量子比特的状态，基于量子门的线性性质，量子门通常表示为矩阵形式.

!!! example

    例如：考虑一个经典的单比特逻辑门 - 非门，非门的操作是将 $0$ 态和 $1$ 态交换。根据非门的定义，我们可以来定义一个量子非门，即把状态

    $$
    \alpha |0⟩ + \beta |1⟩
    $$

    变换到$|0⟩$和$|1⟩$互换角色的新状态
    
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

    再将量子态$\alpha |0⟩ + \beta |1⟩$写成向量模式为
    
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

在QuICT中，我们使用BasicGate类来实现量子门，包括单/多量子比特门和参数/非参数门。
对于 QuICT 中的每个量子门，它都具有以下属性(以CX门为例)：

```python
from QuICT.core.gate import CX

my_CXGate = CX & [0, 1]     # 构建 CX 量子门，控制位为0，目标位为1
```

### 量子门的基础属性
----
```python
my_CXGate.name          # 门的名称
my_CXGate.type          # 门的种类
my_CXGate.precision     # 门的精度
my_CXGate.matrix        # 量子门矩阵
my_CXGate.qasm()        # 量子门的 OpenQASM 字符串
```

### 量子门的比特位和参数信息
----
```python
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
----
```python
my_CXGate.is_single()           # 判断门是否为一量子位门
my_CXGate.is_control_single()   # 判断gate是否有一个控制位和一个目标位
my_CXGate.is_clifford()         # 判断量子门是否属于Clifford
my_CXGate.is_pauli()            # 判断量子门是否属于Pauli
my_CXGate.is_identity()         # 判断量子门矩阵是否为单位矩阵
my_CXGate.is_diagonal()         # 判断门的矩阵是否对角线
my_CXGate.is_special()          # 判断门是否为特殊门，如Measure, Reset, Barrier, Unitary, ...

other_gate = CX & [3, 4]
my_CXGate.commutative(other_gate)   # 当前量子门是否能和目标量子门交换
inverse_gate = my_CXGate.inverse()  # 得到当前量子门的反转量子门
```

### 当前QuICT所支持的量子门种类
|  量子门种类  |   名称   |
|   ------    | ------- |
|    单比特门    | H, HY, S, S_dagger, X, Y, Z, SX, SY, SW, ID, U1, U2, U3, Rx, Ry, Rz, Ri, T, T_dagger, Phase |
|    双比特门    | FSim, Rxx, Ryy, Rzz, Rzx, Swap |
|  受控双比特门  | CZ, CX, CY, CH, CRz, CU1, CU3 |
|  多比特门(>=3) | CCX, CCZ, CCRz, QFT, IQFT, CSwap |
|  特殊量子门    | Measure, Reset, Barrier, Perm, Unitary, Multi-Control Toffoli, uniformly control gate |


## 组合量子门 (Composite Gate)
----
组合量子门, 顾名思义，是多个基础量子门的组合。不同于单一量子门，组合量子门可以实现更加复杂的功能，也通过构建组合量子门这种方式，进一步降低构建量子电路时的复杂性和增加代码的复用性。

QuICT 通过构建 CompositeGate 类，来实现组合量子门的功能，它存储了量子门的列表。它使用 $or$ ( | ) 和 $xor$ ( ^ ) 追加量子门或逆门，并使用 $and$ (&) 重新映射控制量子位。

```python
from QuICT.core.gate import *

# set gate's attributes
cx_gate = CX & [1, 3]   # create CX gate with control qubit 1 and target qubit 3
u2_gate = U2(1, 0)      # create U2 gate with parameters 1 and 0

# create composite gate
cg1 = CompositeGate()

# using default quantum gates
H | cg1(1)                                # append H gate with qubit 1
cx_gate | cg1                             # append pre-defined gates
u2_gate | cg1
QFT.build_gate(3) | cg1([0, 3, 4])        # append QFT composite gate with qubit [0, 3, 4]
U1(1) | cg1(4)                            # append U1 gate with parameters 1 and qubit 4   

# using context to build composite gate
with CompositeGate() as cg_context:
    H & 1
    CX & [1, 3]
    QFT.build_gate(3) & [0, 3, 4]
    U1(1) & 4

# Get QASM of CompositeGate
print(cg_context.qasm())
```

在QuICT中，我们内置了一些常见的组合量子门，例如：QFT、多控Toffoli门、CCRz等量子门组合。
```python
from QuICT.core.gate import QFT, MultiControlToffoli

qft_gate = QFT.build_gate(3)    # 构建 3 比特 QFT 量子门
mct = MultiControlToffoli('no_aux')
mct_gate = mct(5)               # 构建 5 比特多控 toffoli 门，不使用辅助比特
```