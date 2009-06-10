from lulu.base import *
from lulu.connected_region import *

import os.path as _path
_basedir = _path.abspath(_path.join(__file__, '..'))

try:
    import functools as _functools
    import nose as _nose
    test = _functools.partial(_nose.run, 'lulu',
                argv=['', '-v', '--exe', '-w', _basedir])
except:
    raise ImportError("Could not load nose.  Please install using"
                      " `easy_install nose`.")

