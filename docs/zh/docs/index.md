# 文档参考

本教程旨在介绍如何使用mkdocs，以及docs格式。

## 使用mkdocs

关于mkdocs，详细教程可以参考：<https://squidfunk.github.io/mkdocs-material/>

在`docs/zh`启动docs服务器：

```shell
mkdocs serve
```

使用上述命令启动服务器所依赖的Python packages：

```shell
pip install mkdocs
pip install mkdocs-material
pip install mkdocs-glightbox
```

如果有希望加入的功能欢迎进一步讨论。

## QuICT docs格式

### 内容

内容上应该是科普（介绍论文）+QuICT实现（代码）

个人觉得可以稍微参考一下量桨的docs内容结构，例如<https://qml.baidu.com/tutorials/machine-learning/data-encoding-analysis.html>

### 图片

#### 图片存储位置

插入图片需要先在`docs/zh/docs/assets/images`中创建与文档内容部分对应的文件夹，用于存图片。

例如编写`docs/zh/docs/tutorials/algorithm/VQA/QAOA.md`，则需要在`docs/zh/docs/assets/images/tutorials/algorithm/VQA/QAOA`存放图片。当然也可以不必放这么深，图片较少的情况下完全可以合并多个文件夹，命名明确，视实际情况而定即可。

#### 图片插入格式

```markdown
<figure markdown>
![图片名]](图片链接){width="xxx"}
<figcaption>图X：图片描述。</figcaption>
</figure>
```

<figure markdown>
![Max-Cut Result](/assets/images/tutorials/algorithm/VQA/QAOA/maxcut_result.png){width="400"}
<figcaption>图1：最大割结果。</figcaption>
</figure>

（图片明确且直观的情况下，图片描述不必须。通常代码运行结果可以不加图片描述，科普部分的图片大概率需要加描述。）

其余需求参考：<https://squidfunk.github.io/mkdocs-material/reference/images/>

### 链接

1. 直接网址链接：<https://www.google.com/>

    ```markdown
    <网页链接>
    ```

2. 文字链接：[Google](https://www.google.com/)

    ```markdown
    [链接描述](网页链接)
    ```

### 表格

```markdown
|  col1  |   col2  |
| ------ | ------- |
|   1    |   xxx   |
|   2    |   xxx   |
|   3    |   xxx   |
```

|  col1  |   col2  |
| ------ | ------- |
|   1    |   xxx   |
|   2    |   xxx   |
|   3    |   xxx   |

其余需求参考：<https://squidfunk.github.io/mkdocs-material/reference/data-tables/>

### 公式

直接用LaTeX公式即可，如行内公式 $\left | \psi  \right \rangle$ 。

行间公式：

\begin{equation}
P(E) ={n \choose k}p^k (1-p)^{n-k}  \tag{1}
\end{equation}

（LaTeX公式在线编辑器：<https://www.latexlive.com/>）

### code blocks

#### 插入代码

```markdown
    ```语言
    代码内容
    ```
```

```python
if __name__ == '__main__':
    print("Hello world!\n")
```

#### code block多选项卡(其他内容的多选项卡同样适用)

```markdown

=== "Python"

    ```python
    if __name__ == '__main__':
        print("Hello world!\n")
    ```

=== "C++"

    ``` c++
    #include <iostream>

    int main(void) {
      std::cout << "Hello world!" << std::endl;
      return 0;
    }
    ```

```

=== "Python"

    ```python
    if __name__ == '__main__':
        print("Hello world!\n")
    ```

=== "C++"

    ``` c++
    #include <iostream>

    int main(void) {
      std::cout << "Hello world!" << std::endl;
      return 0;
    }
    ```

其余需求参考：<https://squidfunk.github.io/mkdocs-material/reference/code-blocks/>

### 列表

#### 无序

```markdown
- 123456

    * 1
    * 2
    * 3
```

- 123456

    * 1
    * 2
    * 3

#### 有序

```markdown
1. 123456

    1. 1
    2. 2
    3. 3

```

1. 123456

    1. 1
    2. 2
    3. 3

其余需求参考：<https://squidfunk.github.io/mkdocs-material/reference/lists/>

### 信息块

```markdown
!!! note "note内容"
    此处注释xxx
```

!!! note "note内容"
    此处注释xxx

类似的信息块支持多种，如warning，example，info，quote等等。具体参考：<https://squidfunk.github.io/mkdocs-material/reference/admonitions/>

### 其余功能

参考：<https://squidfunk.github.io/mkdocs-material/reference/>

---

## 参考文献

### 引用部分格式

QAOA[<sup>[1]</sup>](#refer1)

``` markdown
QAOA[<sup>[1]</sup>](#refer1)
```

### 文献部分格式

<div id="refer1"></div>
<font size=3>
[1] Farhi, E., Goldstone, J. & Gutmann, S. A Quantum Approximate Optimization Algorithm. [arXiv:1411.4028 (2014)](https://arxiv.org/abs/1411.4028)
</font>

<div id="refer2"></div>
<font size=3>
[2] Farhi, E., Goldstone, J. & Gutmann, S. A Quantum Approximate Optimization Algorithm. [arXiv:1411.4028 (2014)](https://arxiv.org/abs/1411.4028)
</font>

``` markdown
---

## 参考文献

<div id="refer1"></div>
<font size=3>
[1] Farhi, E., Goldstone, J. & Gutmann, S. A Quantum Approximate Optimization Algorithm. [arXiv:1411.4028 (2014)](https://arxiv.org/abs/1411.4028)
</font>

<div id="refer2"></div>
<font size=3>
[2] Farhi, E., Goldstone, J. & Gutmann, S. A Quantum Approximate Optimization Algorithm. [arXiv:1411.4028 (2014)](https://arxiv.org/abs/1411.4028)
</font>

---
```

---