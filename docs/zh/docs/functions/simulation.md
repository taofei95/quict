# 量子模拟
经典计算机上的量子计算模拟
=====================
如何验证量子算法的正确性是量子计算最重要的部分之一。使用经典机器模拟量子电路是验证量子算法的一种方式。在 QuICT 中，Simulator 用于模拟量子电路运行过程中量子比特的状态。

|  模拟器后端   |   CPU   |   GPU   |
| ------      | ------- |  ------  |
|   状态向量    |   &#10004;   |  &#10004;   |
|    酉矩阵     |   &#10004;   |  &#10004;   |
|   密度矩阵    |   &#10004;   |  &#10004;   |
|   分布式      |    &#10008;  |  &#10004;   |

模拟器将返回一个数据结构，该数据结构存储有关量子电路模拟的信息

    - 设备：硬件
    - 后端：模拟器的模式
    - 镜头：模拟的重复次数
    - 选项：模拟器的参数
    - 时间：消费时间
    - 计数：每次模拟的测量结果字典

状态向量模拟器
============
状态向量模拟器在运行量子电路期间保持量子比特的状态。运行后通过给定的量子电路，它返回量子电路的测量。

Example
>>>>>>>

```python
from QuICT.core import Circuit
from QuICT.core.utils import GateType
from QuICT.simulation import Simulator

# Build circuit with 100 random gates and 5 qubits
circuit = Circuit(5)
type_list = [GateType.x, GateType.h, GateType.rx, GateType.ry, GateType.rz, GateType.cx]
circuit.random_append(rand_size=100, typelist=type_list)

# Initial Simulator
simulator = Simulator(
    device="GPU",
    backend="statevector",
    shots=10,
    precision="double"
)
result = simulator.run(circuit)    # get simulation's result
```

```
{'id': '1778fbd88b0911ecb845233b8af251ab',
    'device': 'GPU',
    'backend': 'statevector',
    'shots': 10,
    'options': {'precision': 'double',
    'gpu_device_id': 0,
    'sync': False,
    'optimize': False},
    'spending_time': 0.2988075494766236,
    'output_path': '~/QuICT/example/demo/output/1778fbd88b0911ecb845233b8af251ab',
    'counts': defaultdict(int,
                {'10010': 2,
                '00000': 1,
                '11111': 1,
                '11001': 1,
                '00110': 1,
                '01101': 1,
                '11010': 1,
                '11100': 1,
                '10001': 1})}
```


酉矩阵模拟 (Unitary Simulator)
=========
酉模拟器分为两步来模拟量子电路。首先，它计算给定量子电路的酉矩阵。之后，幺正模拟器使用线性使用酉矩阵计算量子位状态向量点的操作，并使用测量操作生成最终的量子比特状态。

Example
>>>>>>>
```python
# Initial unitary simulator
unitary_simulator = Simulator(
    device="CPU",
    backend="unitary",
    shots=10
)
result = unitary_simulator.run(circuit)    # get simulation's result
```

密度矩阵模拟器
============


其他平台模拟器支持
================
目前，QuICT 支持使用其他平台（Qiskit 和QCompute）的模拟器进行仿真。

> Important: 使用前需要安装对应平台的python软件包，以及平台的远程访问权限。

```python
# Initial remote simulator
simulator = Simulator(
    device="qcompute",
    backend="cloud_baidu_sim2_earth",
    shots=10,
    token=qcompute_token
)
result = simulator.run(circuit)    # get simulation's result
```
