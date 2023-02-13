# 酉矩阵分解

酉矩阵分解 (Unitary Decomposition) 算法针对给定的酉矩阵返回对应的量子电路。应用这一方法原则上可以实现对量子态的任意可容许操作，但其电路规模和深度均较高，请斟酌使用。

## 代码实例

``` python
from QuICT.qcda.synthesis import UnitaryDecomposition

UD = UnitaryDecomposition()
gates, _ = UD.execute(mat)
```

所得`gates`即对应于酉矩阵`mat`的量子电路，这里以`CompositeGate`的形式返回，以下给出了一个随机$SU(8)$矩阵对应的3-qubit量子电路。

<figure markdown>
![合成电路](../../../assets/images/functions/QCDA/ud_0.png)
</figure>

## 参考文献

<div id="refer1"></div>
<font size=3>
[1] Shende, V.V., Bullock, S.S., & Markov, I.L. (2006). Synthesis of quantum-logic circuits. IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, 25, 1000-1010. [https://arxiv.org/abs/quant-ph/0406176](https://arxiv.org/abs/quant-ph/0406176)
</font>

<div id="refer2"></div>
<font size=3>
[2] Drury, B., & Love, P.J. (2008). Constructive quantum Shannon decomposition from Cartan involutions. Journal of Physics A: Mathematical and Theoretical, 41, 395305. [https://arxiv.org/abs/0806.4015](https://arxiv.org/abs/0806.4015)
</font>
