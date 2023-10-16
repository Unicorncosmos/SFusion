import os
import platform
import shutil
import sys
import warnings
from os import path as osp
from setuptools import find_packages, setup

import torch
from torch.utils.cpp_extension import (BuildExtension, CppExtension,
                                       CUDAExtension)
def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=[],
                  extra_args=[],
                  extra_include_path=[]):

    define_macros = []
    extra_compile_args = {'cxx': [] + extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
        ]
        sources += sources_cuda
    else:
        print('Compiling {} without CUDA'.format(name))
        extension = CppExtension
        # raise EnvironmentError('CUDA is required to compile MMDetection!')

    return extension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)





if __name__ == '__main__':
    setup(
        ext_modules=[
            make_cuda_ext(
                name='bev_pool_v2_ext',
                module='/SFusion',
                sources=[
                    '/content/SFusion/src/bev_pool.cpp',
                    '/content/SFusion/src/bev_pool_cuda.cu',
                ],
            ),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
