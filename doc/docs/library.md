<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

规划接下来要添加入库的量子算法与量子化简算法

------

# 量子算法

依赖量子计算机实现其高效性的算法

***

## 验证矩阵的乘积

### 问题描述：

给定三个\\\(n \times n\\)的矩阵\\(A, B, C\\)，验证\\(AB=C\\)

### 时间复杂度：

经典：\\(O(n^2)\\)

量子：\\(O(n^{\frac{5}{3}})\\)

### 参考文献：

https://arxiv.org/abs/quant-ph/0409035

--------

## Grover算法及振幅放大

### 问题描述：

原问题：给定一个有\\(N\\)个输入的oracle，对于一个输入\\(x\\)，其输出为1，对其他输入，输出为0。找出输入\\(x\\)

### 时间复杂度：

经典：\\(\Omega(N)\\)

量子：\\(O(\sqrt{N})\\)

### 参考文献：

https://arxiv.org/abs/quant-ph/9701001

### 推广：

振幅放大：https://arxiv.org/abs/quant-ph/0005055



--------

## 有序搜索

### 问题描述：

给定一个oracle，可访问一个有序的\\(N\\)个数的升序数组。给定一个数\\(x\\)，查找其位置。

### 时间复杂度：

经典：\\(log_2N\\)

量子：\\(0.433log_2N\\)

### 参考文献：

https://arxiv.org/abs/quant-ph/0608161

https://arxiv.org/abs/quant-ph/0703231



--------

## 伪币

### 问题描述：

假设有\\(N\\)个硬币，有\\(k\\)枚是伪造的，真硬币的重量相同，假硬币重量也相同，但真假硬币重量不同。给出一个oracle，可以比较两个硬币子集的重量。

### 时间复杂度：

经典：\\(\Omega(klog(N/k))\\)

量子：\\(O(k^{\frac{1}{4}})\\)

### 参考文献：

https://arxiv.org/abs/1009.0416



--------

## 量子模拟

### 问题描述：

对一个物理上存在的\\(n\\)阶自由度哈密顿量\\(H\\)，时间演化算子\\(e^{-iHt}\\)可用\\(poly(n, t)\\)个门实现。

### 参考文献：

一般类哈密顿量：

Quantum information processing in continuous time.  http://pdfs.semanticscholar.org/2816/2ca9d1c6381cd2d917648753f6d530a24019.pdf

Efficient simulation of quantum systems by quantum computers.  https://arxiv.org/abs/quant-ph/9603026.

Simulations of many-body quantum systems by a quantum computer.  https://arxiv.org/abs/quant-ph/9603028.

Adiabatic Quantum State Generation and Statistical Zero Knowledge. https://arxiv.org/abs/quant-ph/0301023

Efficient quantum algorithms for simulating sparse Hamiltonians. https://arxiv.org/abs/quant-ph/0508139

Hamiltonian simulation using linear combinations of unitary operations.  https://arxiv.org/abs/1202.5822

Exponential improvement in precision for Hamiltonian-evolution simulation.  https://arxiv.org/abs/1308.5424

Exponential improvement in precision for simulating sparse Hamiltonians. https://arxiv.org/abs/1312.1414

Simulating Hamiltonian dynamics with a truncated Taylor series. https://arxiv.org/abs/1412.4687

Hamiltonian simulation with nearly optimal dependence on all parameters.  https://arxiv.org/abs/1501.01715

Quantum simulations of one dimensional quantum systems.  https://arxiv.org/abs/1503.06319

A Trotter-Suzuki approximation for Lie groups with applications to Hamiltonian simulation https://arxiv.org/abs/1512.03416

Optimal Hamiltonian simulation by quantum signal processing. https://arxiv.org/abs/1606.02685

Corrected quantum walk for optimal Hamiltonian simulation. https://arxiv.org/abs/1606.03443

Simulating quantum mechanics on a quantum computer. https://arxiv.org/abs/quant-ph/9701019

Hamiltonian simulation by qubitization.  https://arxiv.org/abs/1610.06546

# 量子电路化简算法

量子电路化简算法，实际上是一些经典算法

-----

## noise-adaptive映射

### 问题描述：

将量子比特映射到实际比特上，给定实际比特的CNOT门的错误率，尽可能使得作用CNOT门更可靠

### 参考文献：

https://arxiv.org/pdf/1901.11054.pdf

