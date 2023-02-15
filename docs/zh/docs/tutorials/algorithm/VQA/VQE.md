# 量子变分本征值求解器

量子变分本征值求解器 (Variational Quantum Eigensolver, VQE) 是一个利用量子变分算法 (Variational Quantum Algorithm, VQA) 求解某一Hermitian矩阵最小本征值的方法。此方法常用于求解如指定分子基态能量等问题。由于这类问题本身基于量子力学提出，经典模拟较为困难，同样基于量子力学的量子模拟常被认为是一种有前景的量子优越性实现方案。

本教程旨在给出一个例子，利用经典机器学习库Pytorch和QuICT中内置的量子近似优化算法模块求解$H_6$链分子的基态能量。

## 算法原理

此方法总体流程概述如下：

1. 通过经典方法获取分子在给定基下哈密顿量的二次量子化形式
2. 通过Jordan-Wigner等encoding方法将哈密顿量转化为量子模拟所需的形式
3. 构建Ansatz电路并用VQE方法优化得到分子基态

### 哈密顿量的二次量子化形式与FermionOperator

### QubitOperator与Encoding方法

### Ansatz构建与VQE方法

## 代码实例

本教程所需数据请参见/example/algorithm/molecular_data文件夹，/example/algorithm/hartree_fock_demo.ipynb给出了可以直接运行的代码样例，可供参考。

### 获取哈密顿量

请注意，这一部分代码基于本教程使用的数据格式，用户通常应当根据自己获取数据的方法与格式自行编写相关函数，获取分子哈密顿量的FermionOperator形式后，再行执行后续步骤。

``` python
# 读取数据
moldir = "./molecular_data"
molfile = moldir + "/H6_sto-3g_singlet_linear_r-1.3.hdf5"
moldata = MolecularData(molfile)

overlap = np.load(moldir + "/overlap.npy")
Hcore = np.load(moldir + "/h_core.npy")
two_electron_integral = np.einsum("psqr", np.load(moldir + "/tei.npy"))  # (1, 1, 0, 0)

_, X = sp.linalg.eigh(Hcore, overlap)
obi = obi_basis_rotation(Hcore, X)
tbi = tbi_basis_rotation(two_electron_integral, X)
molecular_hamiltonian = generate_hamiltonian(moldata.nuclear_repulsion, obi, tbi)

# 转化形式
fermi_op = molecular_hamiltonian.get_fermion_operator()
orbitals = 2 * moldata.n_orbitals
electrons = moldata.n_electrons
qubit_op = JordanWigner(orbitals).encode(fermi_op)
hamiltonian = Hamiltonian(qubit_op.to_hamiltonian())
```

这一部分最终获得的`fermi_op`即$H_6$分子哈密顿量的FermionOperator形式，`qubit_op`为对应的QubitOperator形式，`hamiltonian`则是优化模块所需的哈密顿量。

### 计算基态能量

设定优化相关参数之后即可开始进行基态能量计算

``` python
MAX_ITERS = 1000
LR = 0.1
```

初始化网络、经典优化器以及学习率更新

``` python
hfvqe_net = HartreeFockVQENet(orbitals, electrons, hamiltonian)
optim = torch.optim.Adam([dict(params=hfvqe_net.parameters(), lr=LR)])
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=500, gamma=0.1)
```

开始进行训练

``` python
hfvqe_net.train()
loader = tqdm.trange(MAX_ITERS, desc="Training", leave=True)
for it in loader:
    optim.zero_grad()
    state = hfvqe_net()
    loss = hfvqe_net.loss_func(state)
    loss.backward()
    optim.step()
    scheduler.step()
    loader.set_postfix(loss=loss.item())
```

所得`loss`数值即所求基态能量。
