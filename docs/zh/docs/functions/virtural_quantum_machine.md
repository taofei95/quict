# 量子物理机仿真模型
量子物理机是使用量子位（qubits，量子比特）而非经典比特（bits，位）存储和处理信息的特殊计算机。量子位具有经典比特所没有的特殊性质，例如超位置、量子纠缠和量子干涉。当前量子物理机主要分为:
    - 超导量子物理机
    - 离子阱量子物理机
    - 光学量子物理机
    - 核磁共振量子物理机

目前量子物理机还处于初期发展阶段，受到当前技术和环境影响，量子物理机不可避免的需要面对诸多噪声和误差。QuICT 尝试通过分析量子物理机中的量子比特的特性、量子指令集和拓扑结构，来构造与真实量子物理机类似的噪声模型，从而进行基于量子物理机的仿真模拟。

## 量子比特 （Qubit）
对于每台量子计算机来说，其中的量子比特都具有不同的实验参数，我们主要考虑以下几种：
- T1/T2 相干时间
- 测量/制备保真度
- 量子门保真度
- 工作/测量频率

相关代码请参考 example/core/qubit.py 或者本文档中量子比特部分

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


## 拓扑结构（Topology）
量子物理机中比特拓扑结构是量子计算机中的一种关键结构，其合理性和优化程度会直接影响到量子计算机的运算速度和计算能力。

目前，存在许多种不同的比特排列结构，下面列举几种常见的结构形式。

    - 线性排列结构：线性排列结构最简单，也最基础的一种比特排列方式，其中比特沿着一条线性路径集中排列。这种排列方式不仅易于实现，但同时存在着数量极大的交错路径导致比特之间的耦合受到限制的问题。

    - 矩阵结构：矩阵结构是指比特沿着矩阵状排列，其中每个比特都可以和其周围的比特进行直接的耦合。与线性排列结构相比，矩阵结构可以更有效地缓解比特之间的交错路径的影响，但同时存在着比特数量受限制等问题。

    - 径向排列结构：径向排列结构是指比特集中排列，形成如同“放射线”形的结构。这种排列方式最适合于某些特定的应用场景，例如量子随机游走和量子模拟等。

除了上述常见的排列结构之外，还有其他一些将比特排列成方格和蜂巢状等几何结构的，以此来实现比特之间的经典延迟和耦合。这些排列结构的具体选用，一般要考虑到量子门操作的效率、量子比特之间的经典延迟、量子比特的稳定性、设备的规模和工程的可行性等因素。

### 如何在 QuICT 中构建一个量子比特拓扑结构
``` python
from QuICT.core import Layout, Circuit

# Build a linearly layout with 5 qubits
layout = Layout(qubit_number=5)
layout.add_edge(0, 1, directional=False, error_rate=1.0)
layout.add_edge(1, 2, directional=False, error_rate=1.0)
layout.add_edge(2, 3, directional=False, error_rate=1.0)
layout.add_edge(3, 4, directional=False, error_rate=1.0)
print(layout)
```
相关代码请参考 example/core/layout.py


## 如何构建一个量子物理机模型
通过整合上述量子比特、指令集和拓扑结构信息，我们尝试来搭建量子物理机模型 **VirtualQuantumMachine**。
``` python
from QuICT.core import Layout, Qureg
from QuICT.core.virtual_machine import InstructionSet, VirtualQuantumMachine
from QuICT.core.utils import GateType


qubit_number = 9
# 构建量子比特
qureg = Qureg(qubit_number)
qureg.set_t1_time([30.1] * 9)
qureg.set_fidelity([0.99] * 9)
gate_fidelity = {GateType.h: 0.99, GateType.rx: 0.98, GateType.ry: 0.89, GateType.rz: 0.89}
qureg.set_gate_fidelity(gate_fidelity)
cs = [
    (0, 1, 0.9), (0, 3, 0.9), (1, 2, 0.91), (1, 4, 0.91), (2, 5, 0.8), (3, 4, 0.6),
    (3, 6, 0.6), (4, 5, 0.5), (4, 7, 0.5), (5, 8, 0.45), (6, 7, 0.45), (7, 8, 0.45),
]
qureg.set_coupling_strength = cs

# 构建量子指令集
iset = InstructionSet(GateType.cx, [GateType.h, GateType.rx, GateType.ry, GateType.rz])
# 构建量子比特拓扑结构
layout = Layout.grid_layout(qubit_number=9)

# 构建量子物理机模型
vqm = VirtualQuantumMachine(
    qubits=qureg,
    instruction_set=iset,
    preparation_fidelity=[0.8] * 9,
    layout=layout,
)
```
另外，针对当前已公开的量子物理机，我们内置了诸多量子物理机模型在QuICT中，其中包括：本源、百度、量子院等
``` python
from QuICT.core.virtual_machine.quantum_machine import OriginalKFC6130

vqm = OriginalKFC6130
```


### 生成量子物理机噪声模型
通过量子物理机模型，QuICT 可以根据其中的量子比特的相关保真度、量子门保真度和耦合强度等，将其转换为量子噪声模型，并在量子电路模拟中进行仿真模拟。
``` python
from QuICT.core.noise import NoiseModel

nm = NoiseModel(quantum_machine_info=vqm)
```

### QCDA 电路自动优化
在 QuICT 的 QCDA 架构中，已支持通过输入量子物理机模型和目标电路，将目标电路转换为当前量子物理机可执行的量子电路。

``` python
from QuICT.core import Circuit
from QuICT.qcda import QCDA
from QuICT.core.virtual_machine.quantum_machine import OriginalKFC6130


circuit = Circuit(6)
circuit.random_append(20, random_params=True)

qcda = QCDA()
circuit_phy = qcda.auto_compile(circuit, OriginalKFC6130)
```

