# QAE算法

QAE算法计算一个量子态在目标空间上的振幅。更详细地，输入是期望精度$\epsilon$、oracle电路$S_\chi$（与Grover算法中的输入相同）、状态制备电路$S_0$；以高概率输出振幅估计$\tilde a,|a-\tilde a|<\epsilon$，其中$S_0\ket{x}=\sqrt{a}\ket{\Psi_1}+\sqrt{1-a}\ket{\Psi_0}$，$\ket{\Psi_1}$是归一化的解空间向量。

框架中实现了三个振幅估计问题的算法（canonical QAE[<sup>[1]</sup>](#refer1)、MLAE[<sup>[2]</sup>](#refer2)、FQAE[<sup>[3]</sup>](#refer3)）。查询复杂度上第一个算法最优；而电路宽度上第二个算法最优；第三个算法在电路宽度与第二个算法基本一致（常数差距）的同时有更小的查询复杂度，而且在实际应用中表现较好。

## 基本用法

`QAE`类位于`QuICT.algorithm.quantum_algorithm.amplitude_estimate`。初始化参数包括：

1. `mode`：字符串，可以是`canonical`，`fast`，`max_likely`中的一个
2. `eps`：输出的期望精度。默认为0.1
3. `simulator`：模拟器。默认值`CircuitSimulator()`

`circuit`方法用于输出电路（只在`canonical`模式可用）；`run`方法用于直接执行算法。为了准备算法所需的输入，需要构造`OracleInfo`对象和`StatePreparationInfo`对象（可选，默认为一层H门）。

## 代码示例

以下代码中，目标空间为最后两位为11的状态，状态制备电路为$H^{\otimes n}$，振幅$a=1/4$。

```python
from QuICT.algorithm.quantum_algorithm import QAE, StatePreparationInfo, OracleInfo
from QuICT.core.gate import *
from QuICT.simulation.state_vector import CircuitSimulator


def example_oracle(n):
    def S_chi(n, controlled=False):
        # phase-flip on target
        cgate = CompositeGate()
        if controlled:
            H | cgate(2)
            CCX | cgate([0, 1, 2])
            H | cgate(2)
        else:
            CZ | cgate([0, 1])
        return cgate

    def is_good_state(state_string):
        return state_string[:2] == "11"

    return OracleInfo(n=n, n_ancilla=0, S_chi=S_chi, is_good_state=is_good_state)


n = 3
eps = 0.05
oracle = example_oracle(n)
pr_function_good = 0
for i in range(1 << n):
    if oracle.is_good_state(bin(i)[2:].rjust(n, "0")):
        pr_function_good += 1
pr_function_good /= 1 << n
pr_success = 0
n_sample = 100
for i in range(n_sample):
    pr_quantum_good = QAE(mode="max_likely", eps=eps).run(oracle=oracle)
    print(f"{pr_quantum_good:.3f} from {pr_function_good:.3f}")
    if np.abs(pr_function_good - pr_quantum_good) < eps:
        pr_success += 1
pr_success /= n_sample
print(f"success rate {pr_success:.2f} with {n_sample:4} samples")
```

## 参考文献

<div id="refer1"></div>
<font size=3>
[1] Brassard, G., Høyer, P., Mosca, M., Montreal, A., Aarhus, B.U., & Waterloo, C.U. (2000). Quantum Amplitude Amplification and Estimation. arXiv: Quantum Physics.
</font>

<div id="refer2"></div>
<font size=3>
[2] Suzuki, Y., Uno, S., Putra, R.H., Tanaka, T., Onodera, T., & Yamamoto, N. (2019). Amplitude estimation without phase estimation. Quantum Information Processing, 19, 1-17.
</font>

<div id="refer3"></div>
<font size=3>
[3] Nakaji, K. (2020). Faster amplitude estimation. Quantum Inf. Comput., 20, 1109-1122.
</font>

