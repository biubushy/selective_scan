import os
from setuptools import setup

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    TORCH_AVAILABLE = True
except ImportError:
    BuildExtension = None
    CUDAExtension = None
    TORCH_AVAILABLE = False

def get_cuda_files():
    core_dir = os.path.join(os.path.dirname(__file__), 'core')
    cuda_files = [
        os.path.join(core_dir, 'selective_scan.cpp'),
        os.path.join(core_dir, 'selective_scan_fwd_fp32.cu'),
        os.path.join(core_dir, 'selective_scan_fwd_fp16.cu'),
        os.path.join(core_dir, 'selective_scan_fwd_bf16.cu'),
        os.path.join(core_dir, 'selective_scan_bwd_fp32_real.cu'),
        os.path.join(core_dir, 'selective_scan_bwd_fp32_complex.cu'),
        os.path.join(core_dir, 'selective_scan_bwd_fp16_real.cu'),
        os.path.join(core_dir, 'selective_scan_bwd_fp16_complex.cu'),
        os.path.join(core_dir, 'selective_scan_bwd_bf16_real.cu'),
        os.path.join(core_dir, 'selective_scan_bwd_bf16_complex.cu'),
    ]
    return cuda_files

extra_compile_args = {
    'cxx': ['-O3', '-std=c++17'],
    'nvcc': [
        '-O3',
        '-std=c++17',
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        '-U__CUDA_NO_HALF2_OPERATORS__',
        '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
        '--expt-relaxed-constexpr',
        '--expt-extended-lambda',
        '--use_fast_math',
    ]
}

ext_modules = []
cmdclass = {}

if TORCH_AVAILABLE:
    ext_modules = [
        CUDAExtension(
            name='selective_scan_cuda',
            sources=get_cuda_files(),
            extra_compile_args=extra_compile_args,
            include_dirs=[os.path.join(os.path.dirname(__file__), 'core')]
        )
    ]
    cmdclass = {'build_ext': BuildExtension}

setup(
    name='selective_scan',
    version='0.1.0',
    author='mamba team',
    description='Selective Scan CUDA kernels extracted from Mamba for standalone use',
    packages=['selective_scan'],
    package_dir={'selective_scan': '.'},
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=[
        'torch>=2.0.0',
    ],
    python_requires='>=3.8',
)

