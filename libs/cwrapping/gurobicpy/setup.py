from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(ext_modules = cythonize([Extension("gurobilpsolver", ["gurobilpsolver.pyx"], libraries=['gurobi80'], include_dirs=[numpy.get_include()], extra_compile_args = ["-mmacosx-version-min=10.5"])]))
#setup(ext_modules = cythonize([Extension("gurobilpsolver", ["gurobilpsolver.pyx"], libraries=['gurobi75'], include_dirs=[numpy.get_include()])]))
