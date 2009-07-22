from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy
import sys

if sys.version_info[:2] < (2, 6):
    package_dir = None
else:
    package_dir = {'lulu': ''}

def cext(name):
    return Extension('lulu.%s' % name, ['lulu/%s.pyx' % name],
                     include_dirs=[numpy.get_include()])

setup(name='lulu',
      version='0.9.3',
      description='Fast and efficient implementation of the 2D '
                  'discrete pulse transform (LULU-operator)',
      author='Stefan van der Walt',
      author_email='stefan@sun.ac.za',
      license='GPL',
      url='http://dip.sun.ac.za/~stefan/code/lulu.git',

      # -----

      cmdclass={'build_ext': build_ext},
      ext_modules=[cext('connected_region'),
                   cext('connected_region_handler'),
                   cext('ccomp'),
                   cext('base')],

      package_data={'': ['*.txt', '*.png', '*.jpg']},
      packages=['lulu', 'lulu.tests'],
      package_dir=package_dir,
)


