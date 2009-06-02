__all__ = ['load_image']

import sys
sys.path.insert(0, '..')

import numpy as np
import PIL.Image

import os

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
