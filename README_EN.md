# Selective Scan CUDA

[中文](README.md) | **English**

Selective Scan CUDA operator extracted from [Mamba](https://github.com/state-spaces/mamba) repository, packaged as a standalone Python library.

> **📢 Important Notice**: This project **extracts** the Selective Scan operator code from [Tri Dao](https://github.com/tridao)'s [Mamba](https://github.com/state-spaces/mamba) repository for standalone packaging. We **did not create or modify** the core algorithm. All CUDA implementations come from the original Mamba project. The value of this project lies in providing a lightweight, independently installable packaging solution.

## Project Overview

**Core Work of This Project:** Extract the CUDA core code of the Selective Scan algorithm from [Tri Dao](https://github.com/tridao)'s [Mamba original implementation](https://github.com/state-spaces/mamba), repackage it as an independently installable Python dependency for easy use in other projects.

**Technical Description:** Selective Scan is the core operator of the Mamba architecture, transforming traditional O(n) serial recursion into O(log n) parallelizable operations through parallel scan techniques. The original implementation is deeply integrated into the Mamba project. This project extracts it as an independent module while maintaining the original high-performance CUDA implementation.

**Extraction and Packaging Features:**
- ✅ Completely preserves the original Mamba Selective Scan implementation
- ✅ Independent installation without full Mamba dependencies
- ✅ Standard PyTorch extension interface
- ✅ Maintains original performance optimizations (multi-precision, parallel scan)
- ✅ Supports real/complex numbers, variable parameters, and other original features
- ✅ CUDA/ROCm dual-platform compatibility

---

## Project Positioning

**This is NOT an original implementation**: All CUDA code in this project comes from the Mamba repository. Our work only includes:
1. Extracting Selective Scan related code from the Mamba repository
2. Reorganizing directory structure for independent compilation
3. Writing setup.py and Python interfaces, packaging as a standard library
4. Providing complete documentation for independent use

**Why extraction is needed**: The original Mamba repository contains complete model implementation with large size and complex dependencies. If you only need to use the Selective Scan operator, installing the entire Mamba introduces unnecessary dependencies. This project provides a lightweight independent installation solution.

**Acknowledgments**: All credit for the core algorithm and CUDA implementation goes to [Tri Dao](https://github.com/tridao) and the Mamba team. This project only performs extraction and packaging work.

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.0 or ROCm >= 5.0
- GCC >= 7.0
- NVIDIA GPU (Compute Capability >= 7.0)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/biubushy/selective_scan.git
cd selective_scan
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n selective_scan python=3.10
conda activate selective_scan

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### 3. Install PyTorch

Visit [PyTorch Official Website](https://pytorch.org/) and install according to your CUDA version:

```bash
# Example: CUDA 12.1
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# Or using pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Compile and Install

**Method A: Development Mode (Recommended for development)**

```bash
python setup.py develop
```

**Method B: Standard Installation**

```bash
pip install .
```

**Method C: Direct Install from Source (For use in other projects)**

```bash
# In your project
pip install git+https://github.com/biubushy/selective_scan.git
```

### 5. Verify Installation

```bash
python -c "import selective_scan; print(f'Version: {selective_scan.__version__}')"
```

---

## Quick Start

### Basic Usage

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

### With Automatic Differentiation

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

### Complete Configuration Example

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

## Using in Projects

### Install as Dependency

Add to your project's `requirements.txt`:

```text
torch>=2.0.0
selective_scan @ git+https://github.com/biubushy/selective_scan.git
```

Then install:

```bash
pip install -r requirements.txt
```

### Local Path Installation

If you cloned the source to local:

```bash
pip install /path/to/selective_scan
```

### Integration into Custom Modules

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

## API Documentation

### Main Function

#### `selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False)`

Execute selective scan operation.

**Parameters:**

- `u` (Tensor): Input sequence, shape `[batch, dim, seqlen]`
- `delta` (Tensor): Step size parameter, shape `[batch, dim, seqlen]`
- `A` (Tensor): State transition matrix, shape `[dim, dstate]`, supports real/complex
- `B` (Tensor): Input matrix
  - Fixed mode: `[dim, dstate]`
  - Variable mode: `[batch, n_groups, dstate, seqlen]`
- `C` (Tensor): Output matrix
  - Fixed mode: `[dim, dstate]`
  - Variable mode: `[batch, n_groups, dstate, seqlen]`
- `D` (Tensor, optional): Skip connection, shape `[dim]`
- `z` (Tensor, optional): Gating input, shape `[batch, dim, seqlen]`
- `delta_bias` (Tensor, optional): Delta bias, shape `[dim]`
- `delta_softplus` (bool): Whether to apply softplus activation to delta

**Returns:**

- If no `z`: `out` (Tensor), shape `[batch, dim, seqlen]`
- If has `z`: `(out, out_z)` (Tuple[Tensor, Tensor])

**Example:**

```python
out = selective_scan.selective_scan_fn(u, delta, A, B, C)
```

### Auxiliary Functions

#### `selective_scan_forward(...)`

Directly calls forward CUDA kernel without building autograd graph.

#### `selective_scan_backward(...)`

Directly calls backward CUDA kernel.

---

## Advanced Usage

### Mixed Precision Training

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

### Variable B and C (Time-dependent Parameters)

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

### Complex Weights

```python
u = torch.randn(2, 16, 128, device='cuda', dtype=torch.float32)
delta = torch.randn(2, 16, 128, device='cuda', dtype=torch.float32)
A = torch.randn(16, 16, device='cuda', dtype=torch.complex64)
B = torch.randn(16, 16, device='cuda', dtype=torch.complex64)
C = torch.randn(16, 16, device='cuda', dtype=torch.complex64)

out = selective_scan.selective_scan_fn(u, delta, A, B, C)
```

---

## Performance Optimization Tips

1. **Sequence Length**: Kernel automatically selects optimal thread configuration for different sequence lengths
2. **Batch Size**: Recommend batch sizes as powers of 2 for GPU memory alignment
3. **State Dimension**: `dstate <= 256`, exceeding this causes shared memory overflow
4. **Precision Selection**: 
   - FP32: Highest precision, slower speed
   - FP16: Balanced precision and speed
   - BF16: Good training stability, recommended for large models

---

## Troubleshooting

### Compilation Errors

**Issue**: `No module named 'torch'`

**Solution**: Ensure PyTorch is installed before compiling this project

```bash
pip install torch
python setup.py develop
```

**Issue**: CUDA version mismatch

**Solution**: Ensure PyTorch's CUDA version matches system CUDA version

```bash
python -c "import torch; print(torch.version.cuda)"
nvcc --version
```

### Runtime Errors

**Issue**: `CUDA out of memory`

**Solution**: Reduce batch size or sequence length

**Issue**: `Expected stride to be 1`

**Solution**: Ensure input tensors are contiguous

```python
u = u.contiguous()
delta = delta.contiguous()
```

---

## Project Structure

```
selective_scan/
├── __init__.py              # Python API interface
├── setup.py                 # Installation configuration
├── README.md                # Chinese documentation
├── README_EN.md             # This document
├── .gitignore              # Git ignore configuration
└── core/                   # CUDA core implementation
    ├── selective_scan.cpp           # PyTorch extension entry
    ├── selective_scan.h             # Parameter structure definitions
    ├── selective_scan_common.h      # Common utilities
    ├── selective_scan_fwd_kernel.cuh    # Forward kernel
    ├── selective_scan_bwd_kernel.cuh    # Backward kernel
    ├── selective_scan_fwd_*.cu      # Forward instantiations
    ├── selective_scan_bwd_*.cu      # Backward instantiations
    ├── reverse_scan.cuh             # Reverse scan implementation
    ├── static_switch.h              # Compile-time branching
    └── uninitialized_copy.cuh       # Memory utilities
```

---

## Citation

**Core Algorithm Citation**: If using the Selective Scan algorithm, please cite the original Mamba paper:

```bibtex
@article{mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

**This Extraction Project Citation** (Optional):

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

## License

**Important Notice**:
- The core CUDA code in this project comes entirely from the [Mamba original repository](https://github.com/state-spaces/mamba) ([Tri Dao](https://github.com/tridao) © 2023)
- We only performed code extraction and packaging work, with no modifications to the algorithm implementation
- All code follows the license of the original Mamba project
- Packaging and documentation by biubushy (2025-10)

---

## Contributing

Issues and Pull Requests are welcome!

**Contribution Scope**:
- ✅ Packaging layer improvements (setup.py, __init__.py)
- ✅ Documentation enhancements and example additions
- ✅ Installation script optimization
- ✅ Bug fixes and compatibility improvements
- ❌ Core CUDA algorithm modifications (please submit to [Mamba original repository](https://github.com/state-spaces/mamba))

**Development Process**:
1. Fork this repository
2. Create feature branch (`git checkout -b feature/PackagingImprovement`)
3. Commit changes (`git commit -m 'Improve packaging'`)
4. Push to branch (`git push origin feature/PackagingImprovement`)
5. Open Pull Request

---

## Version History

- **v0.1.0** (2025-10) - Initial extraction version
  - Extracted Selective Scan core code from Mamba repository
  - Reorganized into standalone Python package structure
  - Preserved all original features: forward/backward pass, multi-precision, real/complex support
  - Added standard setup.py and __init__.py interfaces
  - Wrote complete usage documentation

---

## Contact

- GitHub: [@biubushy](https://github.com/biubushy)
- Project Homepage: https://github.com/biubushy/selective_scan
- Issues: https://github.com/biubushy/selective_scan/issues

---

**biubushy | 2025-10**

