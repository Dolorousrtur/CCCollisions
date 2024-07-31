from setuptools import setup, Extension
from torch.utils import cpp_extension
import torch
import os.path as osp

from torch.utils.cpp_extension import CUDAExtension

bvh_include_dirs = torch.utils.cpp_extension.include_paths() + [
      'include', 'include/cgbn',
      osp.expandvars('$CUDA_SAMPLES_INC')]

extra_compile_args = {'nvcc':
                        [],
                          'cxx': []}

bvh_src_files = ['src/cccollisions.cpp', 'src/out.cu']
bvh_extension = CUDAExtension('cccollisions', bvh_src_files,
                              include_dirs=bvh_include_dirs,
                              extra_compile_args=extra_compile_args)

setup(name='cccollisions',
      ext_modules=[bvh_extension],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
