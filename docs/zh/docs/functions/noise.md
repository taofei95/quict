# 量子噪声

在对量子计算进行模拟的时候，我们很多时候只处理封闭量子系统的动力学，即量子系统不会与外界发生任何不必要的相互作用。尽管对于在此理想系统中原则上可实现的信息处理任务可以得出让人着迷的结论，但此观察受到如下事实的影响，即在现实世界中没有完全封闭的系统，除了整个宇宙。真实系统遭受与外界不必要的相互作用。这些不必要的相互作用在量子信息处理系统中显示为噪声。我们需要理解和控制这样的噪声过程，以便构建有用的量子信息处理系统。

QuICT目前支持含噪声量子电路的模拟，可以使量子模拟更加贴近真实的量子计算机，我们可以自定义所需要的噪声模型，也可以使用内置的常见的噪声模型，并将其应用在所需要的逻辑门上，然后通过密度矩阵模拟器来进行噪声量子电路的模拟。


## 量子噪声模型

QuICT 目前支持自定义量子噪声，也实现了三种量子噪声模型，分别是泡利信道、振幅和相位阻尼以及测量噪声。

### 泡利信道

- 比特翻转信道以概率 $1 − \rho$ 从 $|0⟩$ 到 $|1⟩$ (或者倒过来) 翻转一个量子比特。它具有操作元
    $E_0 = \sqrt \rho I = \sqrt \rho \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    \end{bmatrix}\ \ \  and \ \ \ E_1 = \sqrt{1 - \rho} X = \sqrt{1 - \rho} \begin{bmatrix}
    0 & 1\\
    1 & 0\\
    \end{bmatrix}$
- 相位翻转信道具有操作元
    $E_0 = \sqrt \rho I = \sqrt \rho \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    \end{bmatrix}\ \ \  and \ \ \ E_1 = \sqrt{1 - \rho} Z = \sqrt{1-\rho} \begin{bmatrix}
    1 & 0\\
    0 & -1\\
    \end{bmatrix}$
- 比特相位翻转信道具有操作元
    $E_0 = \sqrt \rho I = \sqrt \rho \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    \end{bmatrix}\ \ \  and \ \ \ E_1 = \sqrt{1 - \rho} Y = \sqrt{1 - \rho} \begin{bmatrix}
    0 & -i\\
    i & 0\\
    \end{bmatrix}$
- 退极化信道是量子噪声的一种重要的类型。它表示量子比特有概率 $\rho$ 被一个完全混态 $I / 2$ 所替代，有概率 $1 − \rho$ 是不变的。它（单比特）具有操作元
    ${\sqrt {1 − \frac{3\rho}{4}} I , \frac{\sqrt {\rho}}{2}X, \frac{\sqrt {\rho}}{2}Y, \frac{\sqrt {\rho}}{2}Z}$

### 振幅和相位阻尼

- 振幅阻尼是对能量耗散的描述，即由量子系统的能量损失带来的影响。它具有操作元
    $E_0 = \begin{bmatrix}
    1 & 0\\
    0 & \sqrt{1 - \rho}\\
    \end{bmatrix}
    \ \ \ and  \ \ \ 
    E_1 = \begin{bmatrix}
    0 & \sqrt \rho\\
    0 & 0\\
    \end{bmatrix}$
    > QuICT 同样支持广义振幅阻尼
- 相位阻尼是一种独特的量子力学噪声过程，描述了量子信息损失而没有能量损失。它具有操作元
    $E_0 = \sqrt \rho \begin{bmatrix}
    1 & 0\\
    0 & 1\\
    \end{bmatrix}
    \ \ \ and  \ \ \ 
    E_1 = \sqrt {1 - \rho} \begin{bmatrix}
    1 & 0\\
    0 & -1\\
    \end{bmatrix}$

### 测量噪声

- 测量噪声表示分别以一定概率 $\rho(n|m)$ 来输出真实测量值， 以 $1 - \rho (n|m)$ 概率来输出错误的测量结果。
  
    > 单比特测量噪声模型：
    $\rho = \begin{bmatrix}
        \rho(0|0) & \rho(1|0)\\
        \rho(1|0) & \rho(1|1)\\
        \end{bmatrix} = \begin{bmatrix}
        \rho(测量值为0，输出值为0) & \rho(测量值为0，输出值为1)\\
        \rho(测量值为1，输出值为0) & \rho(测量值为1，输出值为1)\\
        \end{bmatrix}$
    > !!! warning
        $\rho$ 的每一行的和必须为1


## 构建含噪声量子电路

QuICT 通过 NoiseModel 类，来实现构造含噪声量子电路。整个模块大致分为两部分，一部分是将定义好的量子噪声加入噪声模型之中，并可以与量子比特和量子门所绑定；另一部分是通过给入想要加入噪声的量子电路，按照模型内的噪声规则生成含噪声的量子电路。

``` python
from QuICT.core import Circuit
from QuICT.core.gate import *
from QuICT.core.noise import *

# 构建量子电路
circuit = Circuit(5)
H | circuit
CX | circuit([1, 2])
SX | circuit(0)
T | circuit(1)
Swap | circuit([2, 1])
U1(np.pi / 2) | circuit(2)
Rx(np.pi) | circuit(1)
Measure | circuit

# 构建噪声模型
bf_err = BitflipError(0.1)      # 构建比特翻转噪声，翻转概率为0.1
dep_error = DepolarizingError(0.05, num_qubits=1)                   # 构建单比特退极化信道，概率为0.05
bits_err = PauliError([('xy', 0.1), ('zi', 0.9)], num_qubits=2)     # 构建双比特泡利信道噪声
readout_err = ReadoutError(np.array([[0.8, 0.2], [0.2, 0.8]]))      # 构建单比特Readout噪声

nm = NoiseModel()   # 初始化噪声模型
nm.add_noise_for_all_qubits(bf_err, ['h'])          # 添加比特翻转噪声，只针对 H 量子门
nm.add_noise_for_all_qubits(dep_error, ['x', 'y'])  # 添加退极化噪声，针对 X，Y 量子门
nm.add(bits_err, ['cx', 'ch'], [1, 2])              # 添加双比特泡利信道噪声，针对比特位为1，2的 CX，CY 量子门
nm.add_readout_error(single_readout, [1, 3])        # 添加Readout噪声，针对位置为1，3的量子比特

noised_circuit = nm.transpile(circuit)  # 生成含噪声量子电路
```

## 含噪声量子电路模拟

通过密度矩阵模拟器可以进行含噪声量子电路的模拟。下面通过一个简单的例子来说明如何进行含噪声量子电路模拟。

- 构建初始量子电路
  
    ``` python
    from QuICT.core import Circuit
    from QuICT.core.gate import *

    # Build a circuit with qubits 5
    circuit = Circuit(5)

    # add gates
    H | circuit(0)                      # append H gate to all qubits
    for i in range(4):
        CX | circuit([i, i+1])          # append CX gate
    ```

<figure markdown>
![circuit_demo](../../../assets/images/functions/circuit_demo.jpg){:width="500px"}
</figure>


- 针对初始量子电路进行模拟
  
    ``` python
    from QuICT.simulation.state_vector import CircuitSimulator

    # 量子电路模拟
    simulator = CircuitSimulator()
    sv = simulator.run(circuit)
    sample_result = simulator.sample(3000)
    ``` 

    ``` python
    [1484, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1516]
    ```

- 构建噪声模型，并使用密度矩阵进行模拟
  
    ``` python
    from QuICT.simulation.density_matrix import DensityMatrixSimulator
    from QuICT.core.noise import NoiseModel, BitflipError

    # 构建噪声模型
    bf_err = BitflipError(0.05)
    bf2_err = bf_error.tensor(bf_error)
    nm = NoiseModel()
    nm.add_noise_for_all_qubits(bits_err, ['cx'])

    # 含噪声量子电路模拟
    simulator = DensityMatrixSimulator()
    sv = dm_simu.run(circuit, noise_model=nm)
    sample_result = dm_simu.sample(3000)
    ```

    ``` python
    [1046, 57, 54, 56, 49, 6, 4, 59, 57, 3, 3, 14, 13, 4, 5, 95, 94, 8, 8, 5, 5, 3, 9, 56, 58, 6, 5, 59, 66, 52, 57, 984]
    ```
