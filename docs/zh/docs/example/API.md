# API文档注释风格

**整体注释风格遵循[Google注释标准](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/#comments)**

## Class

### 类的定义

定义class时的注释需要包含：

- 对类的描述
- 相关公式：正常LaTex格式即可
- Reference：参考论文名+链接
- Note：需要标注的注意事项
- Args：初始化该类时的参数
- Examples：调用该类的简单示例，包含import,代码和结果展示

!!! note "格式注意"

    关于类的注释内容放在声明和__init__()之间，__init__()部分的注释只需一句话：“Initialize a xxx object.”即可

    注意缩进

    公式和矩阵：LaTex格式，短公式前后加`$`即可，长公式或者矩阵需要`$$...$$`，且在不同行上：
    
    ```
    $$
    \operatorname{ker} f=\{g\in G:f(g)=e_{H}\}{\mbox{.}}
    $$
    ```

    链接：< link >

#### 示例

``` python
class Example:
    """Some descriptions of this class.

    For more detail, please refer to:

    Reference:
        `xxx` <https://arxiv.org/>.

    Note:
        xxxxxxx.

    Args:
        arg1 (TYPE): xxx.
        arg2 (TYPE): xxx.

    Examples:
        >>> from xxx import Example
        >>> example = Example()
        >>> print(example)

        XXXXXX
    """

    def __init__(self, arg1: TYPE, arg1: TYPE):
        """Initialize an Example object."""

```

## attribute

面向用户，也需要注释

#### 示例

``` python
@property
def attribute(self) -> TYPE:
    """Get xxx.

    Returns:
        TYPE: xxx.
    """
    return self._attribute
```

``` python
 @attribute.setter
def attribute(self, attribute):
    """Set xxx."""
    self._attribute = attribute
```

## function

#### 函数定义

定义function时的注释需要包含：

- 对函数的描述
- 相关公式：正常LaTex格式即可
- Reference：参考论文名+链接
- Note：需要标注的注意事项
- Args：初始化该类时的参数
- Returns：返回类型和内容
- Raises：error类型和描述

#### 示例

``` python
def func(self, arg1: TYPE, arg2: TYPE = xxx):
    """Some descriptions of this function.

    For more detail, please refer to:

    Reference:
        `xxx` <https://arxiv.org/>.

    Note:
        xxxxxxx.

    Args:
        arg1 (TYPE): xxx.
        arg2 (TYPE): xxx. Defaults xxx.

    Returns:
        TYPE: xxx.

    Raises:
        xxxError: xxx.
    """

    return xxx
```

## 整体Example

``` python
from typing import Union

import numpy as np

from QuICT.core import Circuit
from QuICT.core.gate import Rxx, Rzz, Variable
from QuICT.tools.exception.algorithm import *

from .ansatz import Ansatz


class CRAML(Ansatz):
    r"""The Color-Readout-Alternating-Mixed-Layer architecture (CRAML) Ansatz for QNN.

    For an image of size $2^n \times 2^n$, the number of qubits is $2n + 1$.

    For more detail, please refer to:

    Reference:
        `Image Compression and Classification Using Qubits and Quantum Deep Learning` <https://arxiv.org/abs/2110.05476>.

    Note:
        Only applicable to FRQI encoding or NEQR for binary images.

    Args:
        n_qubits (int): The number of qubits.
        layers (int): The number of layers.

    Examples:
        >>> from QuICT.algorithm.quantum_machine_learning.ansatz_library import CRAML
        >>> craml = CRAML(3, 1)
        >>> craml_cir = craml.init_circuit()
        >>> craml_cir.draw("command")
                ┌─────────────┐┌─────────────┐
        q_0: |0>┤0            ├┤0            ├─■────────────■───────────
                │             ││  rxx(1.503) │ │            │ZZ(1.2849)
        q_1: |0>┤  rxx(1.503) ├┤1            ├─┼────────────■───────────
                │             │└─────────────┘ │ZZ(1.2849)
        q_2: |0>┤1            ├────────────────■────────────────────────
                └─────────────┘
    """

    def __init__(self, n_qubits: int, layers: int):
        """Initialize a CRADL ansatz object."""
        super(CRAML, self).__init__(n_qubits)
        self._color_qubit = n_qubits - 2
        self._readout = n_qubits - 1
        self._layers = layers

    @property
    def readout(self) -> list[int]:
        """Get the readout qubits.

        Returns:
            list: The list of readout qubits.
        """
        return [self._readout]

    def __str__(self):
        return "CRAML(n_qubits={}, layers={})".format(self._n_qubits, self._layers)

    def init_circuit(self, params: Union[Variable, np.ndarray] = None):
        """Initialize a CRAML ansatz with trainable parameters.

        Args:
            params (Union[Variable, np.ndarray], optional): Initialization parameters. Defaults to None.

        Returns:
            Circuit: The CRAML ansatz.

        Raises:
            AnsatzShapeError: An error occurred defining trainable parameters.
        """
        n_pos_qubits = self._n_qubits - 2
        params = (
            np.random.randn(self._layers, n_pos_qubits * 2)
            if params is None
            else params
        )
        params = Variable(pargs=params) if isinstance(params, np.ndarray) else params
        if params.shape == (self._layers, n_pos_qubits * 2):
            self._params = params
        else:
            raise AnsatzShapeError(
                str(self._layers, n_pos_qubits * 2), str(params.shape)
            )

        circuit = Circuit(self._n_qubits)
        for l in range(self._layers):
            for i, k in zip(range(n_pos_qubits), range(0, n_pos_qubits * 2, 2)):
                Rxx(params[l][k]) | circuit([i, self._readout])
                Rxx(params[l][k]) | circuit([i, self._color_qubit])
                Rzz(params[l][k + 1]) | circuit([i, self._readout])
                Rzz(params[l][k + 1]) | circuit([i, self._color_qubit])

        return circuit

```