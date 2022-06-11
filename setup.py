from setuptools import setup, Extension
import numpy

# Common flags for both release and debug builds.
extra_compile_args = ["-I.", "-msse4.1", "-msse4.2", "-O3", "-ffast-math", "-march=native", "-Wno-comment", "-Wno-reorder"]
extra_link_args = []

module_pyUCM = Extension('pyUCM',
                    sources = ['src/pyUCM.cpp', 'src/confidence_measures_utility.cpp', 'src/confidence_measures.cpp', 'src/DSI.cpp', 'src/box_filter.cpp', 'src/generate_samples.cpp', 'src/sort.cpp', 'src/evaluation.cpp'],
                    include_dirs=[numpy.get_include(), '/home/luca/miniconda3/envs/cvlab/include/opencv4', 'include'],
                    libraries = ['opencv_core', 'opencv_imgproc', 'opencv_imgcodecs'],
                    library_dirs = ['/home/luca/miniconda3/envs/cvlab/lib'],
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                    )

setup(
  name = 'pyUCM',
  version = '1.0',
  ext_modules = [module_pyUCM]
)
