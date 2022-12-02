from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='GNNAdvisor1',
    ext_modules=[
        CUDAExtension(
        name='GNNAdvisor1',
        sources=[   
                    'GNNAdvisor1.cpp',
                    'GNNAdvisor_kernel1.cu'
                ]
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })