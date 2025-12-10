# Selective Scan CUDA

**中文** | [English](README_EN.md)

从[Mamba](https://github.com/state-spaces/mamba)原始仓库提取的Selective Scan CUDA算子，封装为独立Python包。

> **📢 重要声明**：本项目是从[Tri Dao](https://github.com/tridao)的[Mamba](https://github.com/state-spaces/mamba)原始仓库中**提取**Selective Scan算子代码，进行独立封装。我们**没有创造或修改**核心算法，所有CUDA实现均来自原始Mamba项目。本项目的价值在于提供了轻量级、可独立安装的封装方案。

## 项目简介

**本项目的核心工作：** 从[Tri Dao](https://github.com/tridao)的[Mamba原始实现](https://github.com/state-spaces/mamba)中提取Selective Scan算法的CUDA核心代码，重新封装为可独立安装的Python依赖包，方便在其他项目中直接使用。

**提取版本信息：**
- 源仓库：[state-spaces/mamba](https://github.com/state-spaces/mamba)
- 提取自commit：[`d49d7c9`](https://github.com/state-spaces/mamba/commit/d49d7c909b5a9d151b3f4e7e8477e7d20f156860)
- 提取日期：2024-08-30
- 原始版本：Mamba v2.2.6

**技术说明：** Selective Scan是Mamba架构的核心算子，通过并行扫描技术将传统O(n)串行递归转化为O(log n)可并行操作。原始实现深度集成在Mamba项目中，本项目将其提取为独立模块，保持原有的高性能CUDA实现。

**提取封装特性：**
- ✅ 完整保留原始Mamba的Selective Scan实现
- ✅ 独立安装，无需完整Mamba依赖
- ✅ 标准PyTorch扩展接口
- ✅ 保持原有性能优化（多精度、并行扫描）
- ✅ 支持实数/复数、可变参数等原有功能
- ✅ CUDA/ROCm双平台兼容

---

## 项目定位

**这不是原创实现**：本项目的所有CUDA代码均来自Mamba原始仓库，我们的工作仅是：
1. 从Mamba仓库中提取Selective Scan相关代码
2. 重新组织目录结构，使其可独立编译
3. 编写setup.py和Python接口，封装为标准包
4. 提供完整文档，便于独立使用

**为什么需要提取**：Mamba原始仓库包含完整的模型实现，体积较大且依赖复杂。如果只需要使用Selective Scan算子，安装整个Mamba会引入不必要的依赖。本项目提供了轻量级的独立安装方案。

**致谢**：核心算法和CUDA实现的所有功劳归于[Tri Dao](https://github.com/tridao)和Mamba团队，本项目仅做提取和封装工作。

---

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0 或 ROCm >= 5.0
- GCC >= 7.0
- NVIDIA GPU (计算能力 >= 7.0)

---

## 安装流程

### 1. 克隆源码

```bash
git clone https://github.com/biubushy/selective_scan.git
cd selective_scan
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用conda
conda create -n selective_scan python=3.10
conda activate selective_scan

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 3. 安装PyTorch

访问 [PyTorch官网](https://pytorch.org/) 根据您的CUDA版本安装：

```bash
# 示例：CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 或使用pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. 编译安装

**方式A：开发模式（推荐用于开发调试）**

```bash
python setup.py develop
```

**方式B：正式安装**

```bash
pip install .
```

**方式C：从源码直接安装（其他项目使用）**

```bash
# 在您的项目中
pip install git+https://github.com/biubushy/selective_scan.git
```

### 5. 验证安装

```bash
python -c "import selective_scan; print(f'Version: {selective_scan.__version__}')"
```

---

## 快速开始

### 基础使用

```python
import torch
import selective_scan

# biubushy | 2025-10
batch, dim, seqlen, dstate = 2, 16, 128, 16
device = 'cuda'

u = torch.randn(batch, dim, seqlen, device=device)
delta = torch.randn(batch, dim, seqlen, device=device)
A = torch.randn(dim, dstate, device=device)
B = torch.randn(dim, dstate, device=device)
C = torch.randn(dim, dstate, device=device)

out = selective_scan.selective_scan_fn(u, delta, A, B, C)
print(f"Input: {u.shape} -> Output: {out.shape}")
```

### 带自动微分

```python
import torch
import selective_scan

u = torch.randn(2, 16, 128, device='cuda', requires_grad=True)
delta = torch.randn(2, 16, 128, device='cuda', requires_grad=True)
A = torch.randn(16, 16, device='cuda', requires_grad=True)
B = torch.randn(16, 16, device='cuda', requires_grad=True)
C = torch.randn(16, 16, device='cuda', requires_grad=True)

out = selective_scan.selective_scan_fn(u, delta, A, B, C)
loss = out.sum()
loss.backward()

print(f"Gradients computed: u.grad={u.grad is not None}")
```

### 完整配置示例

```python
import torch
import selective_scan

batch, dim, seqlen, dstate = 2, 16, 128, 16

u = torch.randn(batch, dim, seqlen, device='cuda', dtype=torch.float32)
delta = torch.randn(batch, dim, seqlen, device='cuda', dtype=torch.float32)
A = torch.randn(dim, dstate, device='cuda', dtype=torch.float32)
B = torch.randn(dim, dstate, device='cuda', dtype=torch.float32)
C = torch.randn(dim, dstate, device='cuda', dtype=torch.float32)
D = torch.randn(dim, device='cuda', dtype=torch.float32)
z = torch.randn(batch, dim, seqlen, device='cuda', dtype=torch.float32)
delta_bias = torch.randn(dim, device='cuda', dtype=torch.float32)

out = selective_scan.selective_scan_fn(
    u, delta, A, B, C,
    D=D,
    z=z,
    delta_bias=delta_bias,
    delta_softplus=True
)

if isinstance(out, tuple):
    out, out_z = out
    print(f"Output: {out.shape}, Gated output: {out_z.shape}")
else:
    print(f"Output: {out.shape}")
```

---

## 在项目中使用

### 作为依赖安装

在您的项目 `requirements.txt` 中添加：

```text
torch>=2.0.0
selective_scan @ git+https://github.com/biubushy/selective_scan.git
```

然后安装：

```bash
pip install -r requirements.txt
```

### 本地路径安装

如果您克隆了源码到本地：

```bash
pip install /path/to/selective_scan
```

### 集成到自定义模块

```python
import torch
import torch.nn as nn
import selective_scan

class MambaBlock(nn.Module):
    def __init__(self, dim, dstate=16):
        super().__init__()
        self.dim = dim
        self.dstate = dstate
        
        self.delta_proj = nn.Linear(dim, dim)
        self.A = nn.Parameter(torch.randn(dim, dstate))
        self.B = nn.Parameter(torch.randn(dim, dstate))
        self.C = nn.Parameter(torch.randn(dim, dstate))
        self.D = nn.Parameter(torch.randn(dim))
    
    def forward(self, x):
        batch, seqlen, dim = x.shape
        x = x.transpose(1, 2)
        
        delta = self.delta_proj(x.transpose(1, 2)).transpose(1, 2)
        
        out = selective_scan.selective_scan_fn(
            x, delta, self.A, self.B, self.C, D=self.D
        )
        
        return out.transpose(1, 2)

model = MambaBlock(dim=64).cuda()
x = torch.randn(2, 100, 64).cuda()
y = model(x)
print(f"Input: {x.shape} -> Output: {y.shape}")
```

---

## API 文档

### 主函数

#### `selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False)`

执行选择性扫描操作。

**参数：**

- `u` (Tensor): 输入序列，形状 `[batch, dim, seqlen]`
- `delta` (Tensor): 步长参数，形状 `[batch, dim, seqlen]`
- `A` (Tensor): 状态转移矩阵，形状 `[dim, dstate]`，支持实数/复数
- `B` (Tensor): 输入矩阵
  - 固定模式: `[dim, dstate]`
  - 可变模式: `[batch, n_groups, dstate, seqlen]`
- `C` (Tensor): 输出矩阵
  - 固定模式: `[dim, dstate]`
  - 可变模式: `[batch, n_groups, dstate, seqlen]`
- `D` (Tensor, 可选): 跳跃连接，形状 `[dim]`
- `z` (Tensor, 可选): 门控输入，形状 `[batch, dim, seqlen]`
- `delta_bias` (Tensor, 可选): delta偏置，形状 `[dim]`
- `delta_softplus` (bool): 是否对delta应用softplus激活

**返回：**

- 如果无`z`: `out` (Tensor)，形状 `[batch, dim, seqlen]`
- 如果有`z`: `(out, out_z)` (Tuple[Tensor, Tensor])

**示例：**

```python
out = selective_scan.selective_scan_fn(u, delta, A, B, C)
```

### 辅助函数

#### `selective_scan_forward(...)`

直接调用前向传播CUDA kernel，不建立自动微分图。

#### `selective_scan_backward(...)`

直接调用反向传播CUDA kernel。

---

## 高级用法

### 多精度训练

```python
import torch
import selective_scan

with torch.cuda.amp.autocast():
    u = torch.randn(2, 16, 128, device='cuda', dtype=torch.float16)
    delta = torch.randn(2, 16, 128, device='cuda', dtype=torch.float16)
    A = torch.randn(16, 16, device='cuda', dtype=torch.float32)
    B = torch.randn(16, 16, device='cuda', dtype=torch.float32)
    C = torch.randn(16, 16, device='cuda', dtype=torch.float32)
    
    out = selective_scan.selective_scan_fn(u, delta, A, B, C)
```

### 可变B和C（时间依赖参数）

```python
batch, dim, seqlen, dstate = 2, 16, 128, 16
n_groups = 4

u = torch.randn(batch, dim, seqlen, device='cuda')
delta = torch.randn(batch, dim, seqlen, device='cuda')
A = torch.randn(dim, dstate, device='cuda')
B_var = torch.randn(batch, n_groups, dstate, seqlen, device='cuda')
C_var = torch.randn(batch, n_groups, dstate, seqlen, device='cuda')

out = selective_scan.selective_scan_fn(u, delta, A, B_var, C_var)
```

### 复数权重

```python
u = torch.randn(2, 16, 128, device='cuda', dtype=torch.float32)
delta = torch.randn(2, 16, 128, device='cuda', dtype=torch.float32)
A = torch.randn(16, 16, device='cuda', dtype=torch.complex64)
B = torch.randn(16, 16, device='cuda', dtype=torch.complex64)
C = torch.randn(16, 16, device='cuda', dtype=torch.complex64)

out = selective_scan.selective_scan_fn(u, delta, A, B, C)
```

---

## 性能优化建议

1. **序列长度**: 针对不同序列长度，kernel会自动选择最优的线程配置
2. **批次大小**: 建议批次大小为2的幂次，利于GPU内存对齐
3. **状态维度**: `dstate <= 256`，超过此值会导致共享内存溢出
4. **精度选择**: 
   - FP32: 最高精度，速度较慢
   - FP16: 平衡精度和速度
   - BF16: 训练稳定性好，推荐用于大模型

---

## 故障排除

### 编译错误

**问题**: `No module named 'torch'`

**解决**: 确保先安装PyTorch再编译本项目

```bash
pip install torch
python setup.py develop
```

**问题**: CUDA版本不匹配

**解决**: 确保PyTorch的CUDA版本与系统CUDA版本一致

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### 运行时错误

**问题**: `CUDA out of memory`

**解决**: 减小批次大小或序列长度

**问题**: `Expected stride to be 1`

**解决**: 确保输入张量是连续的

```python
u = u.contiguous()
delta = delta.contiguous()
```

---

## 项目结构

```
selective_scan/
├── __init__.py              # Python API接口
├── setup.py                 # 安装配置
├── README.md                # 本文档
├── .gitignore              # Git忽略配置
└── core/                   # CUDA核心实现
    ├── selective_scan.cpp           # PyTorch扩展入口
    ├── selective_scan.h             # 参数结构定义
    ├── selective_scan_common.h      # 通用工具
    ├── selective_scan_fwd_kernel.cuh    # 前向kernel
    ├── selective_scan_bwd_kernel.cuh    # 反向kernel
    ├── selective_scan_fwd_*.cu      # 前向实例化
    ├── selective_scan_bwd_*.cu      # 反向实例化
    ├── reverse_scan.cuh             # 反向扫描实现
    ├── static_switch.h              # 编译期分支
    └── uninitialized_copy.cuh       # 内存工具
```

---

## 引用

**核心算法引用**：如果使用了Selective Scan算法，请引用Mamba原始论文：

```bibtex
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

**本提取项目引用**（可选）：

```bibtex
@software{selective_scan_extracted,
  author = {biubushy},
  title = {Selective Scan CUDA: Extracted from Mamba for Standalone Use},
  year = {2025},
  url = {https://github.com/biubushy/selective_scan},
  note = {Extracted and packaged from the original Mamba implementation}
}
```

---

## 许可证

**重要说明**：
- 本项目的核心CUDA代码完全来自[Mamba原始仓库](https://github.com/state-spaces/mamba)（[Tri Dao](https://github.com/tridao) © 2023）
- 我们仅进行了代码提取和封装工作，未对算法实现做任何修改
- 所有代码遵循Mamba原始项目的许可证
- 封装和文档部分由biubushy完成（2025-10）

---

## 贡献

欢迎提交Issue和Pull Request！

**贡献范围**：
- ✅ 封装层改进（setup.py、__init__.py）
- ✅ 文档完善和示例补充
- ✅ 安装脚本优化
- ✅ Bug修复和兼容性改进
- ❌ 核心CUDA算法修改（请向[Mamba原始仓库](https://github.com/state-spaces/mamba)提交）

**开发流程**：
1. Fork本仓库
2. 创建特性分支 (`git checkout -b feature/PackagingImprovement`)
3. 提交更改 (`git commit -m 'Improve packaging'`)
4. 推送到分支 (`git push origin feature/PackagingImprovement`)
5. 开启Pull Request

---

## 版本历史

- **v0.1.0** (2025-10) - 初始提取版本
  - 从Mamba原始仓库提取Selective Scan核心代码
    - 源自commit: `d49d7c909b5a9d151b3f4e7e8477e7d20f156860` (2024-08-30)
    - 对应Mamba v2.2.6版本
  - 重新组织为独立Python包结构
  - 保留原有全部功能：前向/反向传播、多精度、实数/复数支持
  - 添加标准setup.py和__init__.py接口
  - 编写完整使用文档

---

## 联系方式

- GitHub: [@biubushy](https://github.com/biubushy)
- 项目主页: https://github.com/biubushy/selective_scan
- Issues: https://github.com/biubushy/selective_scan/issues

---

**biubushy | 2025-10**

