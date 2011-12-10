"""
Illustrate how the impulse strength varies across the image.

"""

from demo import load_image

from viewer import BaseViewer

import lulu
import lulu.connected_region_handler as crh

from traits.api import HasTraits, Instance, Array, Int, Range, \
                                 on_trait_change, Dict, Bool, Button, \
                                 File
from traitsui.api import Item, View, Group, HGroup, RangeEditor
from traitsui.message import message
from enable.api import Component, ComponentEditor

import numpy as np

class StrengthViewer(BaseViewer):
    pulse_strength = Array

    def default_traits_view(self):
        return View(Group(Item('reconstruction', editor=ComponentEditor()),
                             show_labels=False,
                             show_left=False),
                    Item('threshold_min'),
                    Item('threshold_max'),
                    HGroup(Item('save_file', show_label=False),
                           Item('save_button', show_label=False)),

                    width=800, height=600,
                    resizable=True,
                    title='DPT 2D reconstruction')

    def __init__(self, **kwargs):
        HasTraits.__init__(self, **kwargs)

        self.pulse_strength = np.zeros_like(self.image)
        for area in self.pulses:
            for cr in self.pulses[area]:
                crh.set_array(self.pulse_strength, cr, 1, 'add')

        self.result = self.pulse_strength.copy()

        tmin = int(self.pulse_strength.min())
        tmax = int(self.pulse_strength.max())

        self.add_trait('threshold_min',
                       Range(value=0, low=tmin, high=tmax, mode='slider'))
        self.add_trait('threshold_max',
                       Range(value=tmax, low=tmin, high=tmax, mode='slider'))

    @on_trait_change('threshold_min, threshold_max')
    def reconstruct(self):
        self.result = self.pulse_strength.copy()

        mask = (self.result < self.threshold_min) | \
               (self.result > self.threshold_max)

        self.result[mask] = 0

        self.plot_data.set_data('reconstruction', self.result)
        self.new.request_redraw()

image = load_image()
pulses = lulu.decompose(image)

viewer = StrengthViewer(pulses=pulses, image=image)
viewer.configure_traits()
