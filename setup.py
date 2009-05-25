from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

def cext(name):
    return Extension('lulu.%s' % name, ['lulu/%s.pyx' % name],
                     include_dirs=[numpy.get_include()])

setup(cmdclass = {'build_ext': build_ext},
      ext_modules = [cext('connected_region'),
                     cext('connected_region_handler'),
                     cext('ccomp'),
                     cext('base')])
