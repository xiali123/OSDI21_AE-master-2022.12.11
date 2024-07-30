from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='DCGG',
    ext_modules=[
        CUDAExtension(
        name='DCGG',
        sources=[   
                    'DCGG.cpp',
                    'DCGG_kernel1.cu'
                ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    })