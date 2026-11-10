from pathlib import Path
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

HERE = Path(__file__).parent
setup(
    name='analytic-gs-operators', version='1.0.0',
    ext_modules=[
        CUDAExtension('analytic_anchor_energy',
            [str(HERE/'anchor/bindings.cpp'), str(HERE/'anchor/group_energy.cu')],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2', '-Xptxas=-v']}),
        CUDAExtension('analytic_slot_moments',
            [str(HERE/'slot/bindings.cpp'), str(HERE/'slot/analytic_slots.cu')],
            extra_compile_args={'cxx': ['-O2'], 'nvcc': ['-O2']}),
    ],
    cmdclass={'build_ext': BuildExtension},
)
