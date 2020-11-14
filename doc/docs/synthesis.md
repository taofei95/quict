<script type="text/javascript" src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

# 电路合成

在这里，我们描述QuICT电路合成库中的算法，即将抽象结构分解到门级的算法

***

## Solovay-Kitaev 算法 

### 算法描述

将一个\\(n\\)比特的任意量子电路对应的酉矩阵用一系列固定的\\(n\\)比特量子门进行逼近

目前，由于计算资源限制，平台只支持单比特量子电路的SK分解

#### 参考文献

DAWSON C M, NIELSEN M A. The solovay-kitaev algorithm[J]. arXiv preprint quant-ph/0505030, 2005.

#### 符号定义与相关定理

定义\\(n\\)维量子堆(\\(qudit\\))上的量子门的有限集合\\(\mathcal{G}\\)，称为指令集(\\(instruction set\\))：

1. 对所有\\(g \in \mathcal{G}\\)，\\(g\\)都在\\(SU(n)\\)中，其特征值为1
2. 对所有\\(g \in \mathcal{G}\\)，\\(g^{\dagger}\\)也在\\(\mathcal{G}\\)中
3. \\(\mathcal{G}\\)是\\(SU(d)\\)中的一个宇集(universal set)，即\\(\mathcal{G}\\)生成的群在\\(SU(d)\\)中是稠密的

定义两个酉矩阵\\(U\\)，\\(S\\)的范数距离\\(\Vert U - S \Vert = sup_{\Vert \varphi \Vert = 1} \Vert (U - S)\varphi \Vert \\) 

Solovay-Kitaev定理：对于\\(SU(n)\\)中的一个指令集\\(\mathcal{G}\\)，给定一个精确度\\(\epsilon\\)。存在一个常数c，对于任意\\(U \in SU(n)\\)，有一个由\\(\mathcal{G}\\)中的量子门组成的长度为\\(O(log^c(\frac{1}{\epsilon}))\\)的序列\\(S\\)，\\(U\\)和\\(S\\)的范数距离小于\\(\epsilon\)。

#### 算法输入

一个\\(2^n \times 2^n\\)的酉矩阵

#### 算法输出

一个包含若干个n位量子门的数组

#### 算法实现

1. 设定\\(l_0\\)，枚举长度为\\(l_0\\)的量子门组合\\(\mathcal{G} _ {basic}\\)，同时计算\\(\epsilon_{0}, \epsilon_{0} \geq \min_{S \in \mathcal{G} _ {basic}}sup_{U \in SU(n)}\Vert S - U \Vert  \\\)。在平台实现单比特量子电路SK分解时，使用的指令集包括\\(H, Rz(\pi / 8),  Rz(-\pi / 8)\\)，\\(l_0 = 16\\)，\\(\epsilon_{0} = 0.14\\)

2. 定义\\(GC-Approx-Decompose(U)\\)，返回两个酉矩阵\\(V, W\\)，使得\\(VWV^{\dagger}W^{\dagger}\\)近似\\(U\\):<br/>(1)设\\(U\\)为\\(d \times d\\)的酉矩阵，计算\\(H = -i\log{U}\\)(使用特征值分解)，\\(G = diag(-\frac{d-1}{2}, -\frac{d-1}{2} + 1, \cdots, \frac{d-1}{2})\\)<br/>(2)计算\\(F\\)，$$ F_{jk} = \begin{cases} \frac{iH_{jk}}{G_{kk} - G_{jj}} & \mbox{if \\(j \\neq k\\)} \\\\ 0  &  \mbox{if \\(j = k\\)} \end{cases} $$<br/>(3)调整\\(F\\)和\\(G\\)的特征值的模一致，具体做法为计算\\(f = det(F), g = det(G)\\)，\\(F = \sqrt{\frac{g}{f}}F, G = \sqrt{\frac{f}{g}}G \\)<br/>(4)计算\\(V = e^{iF}, W = e^{iG}\\)<br/>

   获取最坏的近似值\\(c_{approx}\\)，在单比特情况下，\\(c_{approx} < \frac{1}{\sqrt{2}}\\)，在运算中将\\(c_{approx} 取为 \frac{1}{\sqrt{2}}\\)

3. 计算$$ n = \lceil \frac{\ln{ \left [ \ln(1/\epsilon c_{approx}^2) / \ln(1/\epsilon_0 c_{approx}^2 \right ] }}{\ln(3/2)} \rceil $$

4. 调用递归函数\\(Solovay-Kitaev(U, n)\\):<br/>(1)若\\(n = 0\\)，遍历\\(\mathcal{G} _ {basic}\\)，返回与\\(U\\)范数距离最近的组合<br/>(2)令\\(U_{n-1} = Solovay-Kitaev(U, n - 1)\\)  <br/>(3)令\\(V, W = GC-Approx-Decompose(UU_{n-1}^{\dagger}) \\)<br/>(4)令\\(V_{n-1} = Solovay-Kitaev(V, n - 1)\\)<br/>(5)令\\(W_{n-1} = Solovay-Kitaev(W, n - 1)\\)<br/>(6)返回\\(V_{n-1}W_{n-1}V_{n-1}^{\dagger}W_{n-1}^{\dagger}U_{n-1}\\)

#### 算法分析

1. 根据递归函数的性质，序列长度为\\(5^n\\)，时间复杂度为\\(3^n\\)（共轭转置信息可以折叠起来，使得时间复杂度可以短于序列长度，但在实际情况中，由于输出了每个门，时间复杂度为\\(5^n\\)），代入单比特情况与设定指令集的参数，长度为\\(O(ln^{ln 5 / ln(3/2)}(1/\epsilon))\\)，时间为\\(O(ln^{ln 3 / ln(3/2)}(1/\epsilon))\\)，即时间复杂度约为\\(O(log^{2.71}(\frac{1}{\epsilon}))\\)，序列长度约为\\(O(log^{3.71}(\frac{1}{\epsilon}))\\)
2. 在实际运行中，常数\\(3^{16}\\)是运行较慢的主要原因（3为指令集元素个数，16为\\(l_0\\)）

## Quantum Shannon Decomposition算法

### 算法描述

使用Cartan decomposition，将一个n位的任意酉矩阵递归分解为\\(O(4^n)\\)个\\(CNOT\\)门和\\(O(4^n)\\)个单比特门

#### 参考文献

1. Childs A M, Haselgrove H L, Nielsen M A. Lower bounds on the complexity of simulating quantum gates[J]. Physical Review A, 2003, 68(5): 052311.
2. Vatan F, Williams C. Optimal quantum circuits for general two-qubit gates[J]. Physical Review A, 2004, 69(3): 032315.
3. Nakajima Y, Kawano Y, Sekigawa H. A new algorithm for producing quantum circuits using KAK decompositions[J]. arXiv preprint quant-ph/0509196, 2005.
4. Mottonen M, Vartiainen J J. Decompositions of general quantum gates[J]. arXiv preprint quant-ph/0504100, 2005.
5. Drury B, Love P. Constructive quantum Shannon decomposition from Cartan involutions[J]. Journal of Physics A: Mathematical and Theoretical, 2008, 41(39): 395305.

#### 算法输入

一个\\(2^n \times 2^n\\)的酉矩阵

#### 算法输出

n位的量子电路，只包含\\(CNOT\\)门和单量子比特门

#### 算法描述

定义递归函数\\(NKS(U)\\)，它返回对\\(U\\)进行Quantum Shannon Decomposition分解产生的量子门序列：

1. U为\\(4 \times 4\\)酉矩阵时，使用Cartan decomposition \\(\textbf{AI}\\)型进行特殊分解：<br/>(1)令\\(B = \frac{1}{\sqrt{2}} \left( \begin{matrix} 1& i & 0 & 0 \\\\ 0 & 0 & i & 1 \\\\ 0 & 0 & i & -1 \\\\ 1 & -i & 0 & 0 \end{matrix} \right) \\)，\\(U^{'} = B^{\dagger}UB \\)<br/>(2)令\\(M2 = (U^{'})^TU^{'}\\)，并将其对角化为\\(PDP^{\dagger}\\)(\\(P \in SO(4)\\))<br/>(3)计算\\(K^{'} = U^{'}PD^{-\frac{1}{2}}P^{\dagger}, K_1 = BK^{'}PB^{\dagger}, K_2 = BP^{\dagger}B^{\dagger}, A = BD^{\frac{1}{2}}B^{\dagger} \\)，则\\(U = K_1AK_2\\)<br/>(4)分解\\(K_1, K_2 \in SU(2) \otimes SU(2) \\)<br/>(5)分解\\(A \in exp(span_{\mathbb{R}}i\\{XX, YY, ZZ\\})\\)，产生3个\\(CNOT\\)门和4个单比特门，分解方法见参考文献1和2
2. 令\\(\Theta(U^{\dagger}) = Z^nU^{\dagger}Z^n\\)，\\(Z^n = I^{\otimes (n-1)} \otimes Z \\)
3. 令\\(M2 = \Theta(U^{\dagger})U\\)，将其对角化为\\(pbp^{\dagger}\\)，对角化方法见参考文献3的3.2.1
4. 计算\\(y = \sqrt{b}, m = pyp^{\dagger}, k = Um^{\dagger}, k^{'} = kp\\)，则\\(U =  k^{'}yp^{\dagger}\\)
5. 分解\\(k^{'}, p^{\dagger}\\)，它们都有\\(g0 \otimes |0\rangle \langle0| + g1 \otimes |1\rangle \langle1| \\)的形式，这个形式可以分解成两个\\(n-1\\)量子门和\\(uniformlyRz\\)门，分解方法见参考文献5
   1. y则对应一个\\(uniformlyRx\\)门，分解方法见参考文献5

## VBE算法

### 算法描述

待补充

#### 参考文献

待补充

#### 算法输入

待补充

#### 算法输出

待补充

#### 算法描述

待补充

## QCNF算法

### 算法描述

待补充

#### 参考文献

待补充

#### 算法输入

待补充

#### 算法输出

待补充

#### 算法描述

待补充

## MCT算法initial_state_preparation

### 算法描述

（使用辅助比特）对Toffoli门进行门级分解

#### 算法效果

|   名称   | 参数个数 |        额外说明         |
| :------: | :------: | :---------------------: |
| MCT_Linear_Simulation  |    1    |MCT_Linear_Simulation(m) \| qureg，表示一个m控制位的toffoli门 的实现，qureg的前m位作为控制位，最后一位作为作用位，中间的位辅助比特，m不能超过 \\(\\lceil n/2 \\rceil \\)|
| MCT_one_aux_model |0|MCT_one_aux_model \| qureg，qureg的前n位为n-1控制位的toffoli门，最后一位为辅助比特|

## initial_state_preparation算法

### 算法描述

初态制备

#### 算法作用

