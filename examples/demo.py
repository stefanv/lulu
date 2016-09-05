"""\
==============================================
Discrete Pulse Transform / LULU Operator in 2D
==============================================

Demonstrations
--------------
%s

Usage
-----
python demoname.py [image]

If no image file is specified, the demonstration will be run
on Chelsea the Cat.
"""

__all__ = ['load_image']

from glob import glob
import os

__doc__ = __doc__ % \
          '\n'.join(sorted([f[:-3] for f in
                            glob(os.path.dirname(__file__) + '*.py')
                            if f != 'demo.py' and not f.startswith('_')]))

import sys
sys.path.insert(0, '..')

import numpy as np
import PIL.Image

def load_image(fname=None):
    """Load example image.

    """
    if fname is None:
        fname = 'chelsea.jpg'

    fname = os.path.join(os.path.dirname(__file__), 'data/' + fname)

    if len(sys.argv) == 2:
        img = PIL.Image.open(sys.argv[1])
    else:
        img = PIL.Image.open(fname)
    return np.array(img.convert('F')).astype(int)

if __name__ == "__main__":
    print(__doc__)
