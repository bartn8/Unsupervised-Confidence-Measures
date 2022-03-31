from setuptools import setup, Extension
import numpy

# Common flags for both release and debug builds.
extra_compile_args = ["-O3", "-ffast-math", "-march=native", "-Wno-comment", "-Wno-reorder"]
extra_link_args = []

module_pyUCM = Extension('pyUCM',
                    sources = ['src/main.cpp'],
                    include_dirs=[numpy.get_include(), 'include', '/home/luca/miniconda3/envs/cvlab/include/opencv4'],
                    library_dirs = ['/home/luca/miniconda3/envs/cvlab/lib/'],
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                    language = 'c++'
                    )

setup(
  name = 'pyUCM',
  version = '1.0',
  ext_modules = [module_pyUCM],
)
