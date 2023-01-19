# Shor因子分解算法

Shor因子分解算法相对最好的经典算法实现了指数加速，在$O(n^3)$的时间内以高概率给出输入的非平凡因子（如果有）。本框架实现了四个Shor因数分解算法的变体（根据量子部分中使用的乘幂电路/是否使用iterative QPE），在电路宽度与深度上有常数上的差别。

下列表格中，$n$是输入数的位数，$t$是求阶算法中QPE的精度位数。默认$t=2n+1$。

| 算法    | 电路宽度 | 电路深度   | 电路规模   | 模拟器上的运行速度 |
| ------- | -------- | ---------- | ---------- | ------------------ |
| BEA     | 2n+2+t   | $O(n^2 t)$ | $O(n^3 t)$ | 快                 |
| HRS     | 2n+1+t   | $O(n^2 t)$ | $O(n^3 t)$ | 快                 |
| BEA-zip | 2n+3     | $O(n^2 t)$ | $O(n^3 t)$ | 慢                 |
| HRS-zip | 2n+2     | $O(n^2 t)$ | $O(n^3 t)$ | 慢                 |

Shor因子分解算法包含了经典部分（素数判定、求最大公约数等等）与量子部分（求阶算法）。如果在量子部分，对于待分解数$N$，我们使用数$a$满足$(a,N)=1,\text{ord}_N(a)=r,2|r,a^{r/2}\neq -1\bmod N$，那么$\gcd(a^{r/2}-1,N),\gcd(a^{r/2}+1,N)$中必然包含$N$的非平凡因子。更详细的成功概率分析可以参考教科书。我们只考虑对于量子部分的时间开销，对于$n=11$的输入，求阶算法在单块GPU上可以在一小时内完成。

## 基本用法

`ShorFactor`类位于`QuICT.algorithm.quantum_algorithm.shor`，初始化参数包括

1. `mode`：字符串，可以指定为`BEA`[<sup>[1]</sup>](#refer1)、`HRS`[<sup>[2]</sup>](#refer2)、`BEA_zip`、`HRS_zip`中的一个。`*_zip`指使用了iterative QPE[<sup>[3]</sup>](#refer3)（也就是原论文中所说的one-bit trick）
2. `eps`：相位估计的精度
3. `max_rd`：order-finding子程序的最大可执行次数。默认为2
4. `simulator`：模拟器。默认为`CircuitSimulator()`

调用`circuit`方法可以得到order-finding部分的电路；调用`run`方法可以直接执行整个算法。

## 代码示例

参考`example/demo/tutorial_shor.ipynb`。

## 参考文献

<div id="refer1"></div>

<font size=3>
[1] Beauregard, S. (2002). Circuit for Shor's algorithm using 2n+3 qubits. Quantum Inf. Comput., 3, 175-185.
</font>

<div id="refer2"></div>

<font size=3>
[2] Häner, T., Rötteler, M., & Svore, K.M. (2016). Factoring using $2n+2$ qubits with Toffoli based modular multiplication. ArXiv, abs/1611.07995.
</font>

<div id="refer3"></div>

<font size=3>
[3] Griffiths, & Niu (1995). Semiclassical Fourier transform for quantum computation. Physical review letters, 76 17, 3228-3231 .
</font>	