from enthought.enable.api import Component, ComponentEditor
from enthought.traits.api import HasTraits, Instance, Array, Int, Range, \
                                 on_trait_change, Dict
from enthought.traits.ui.api import Item, View, Group, RangeEditor
from enthought.chaco.api import Plot, ArrayPlotData, PlotLabel, \
                                HPlotContainer, gray

from _default_override import DefaultOverride

import numpy as np

from demo import load_image

import lulu
import lulu.connected_region_handler as crh

class Viewer(HasTraits):
    pulses = Dict
    pulses_used = Int
    reconstruction = Instance(Component)

    image = Array
    result = Array

    # Thresholds are defined in __init__

    no_endlabel = DefaultOverride(low_label='', high_label='', mode='slider')

    traits_view = View(Group(Item('reconstruction', editor=ComponentEditor()),
                             show_labels=False,
                             show_left=False),
                       Item('amplitude_threshold_min', editor=no_endlabel,
                            label='Minimum absolute amplitude'),
                       Item('amplitude_threshold_max', editor=no_endlabel),
                       Item('area_threshold_min', editor=no_endlabel),
                       Item('area_threshold_max', editor=no_endlabel),
                       Item('volume_threshold_min', editor=no_endlabel),
                       Item('volume_threshold_max', editor=
                            DefaultOverride(
                                low=1, low_label='', high_label='',
                                mode='logslider')
                            ),

                       width=800, height=600,
                       resizable=True,
                       title='DPT 2D reconstruction')

    def __init__(self, **kwargs):
        HasTraits.__init__(self, **kwargs)

        # Calculate maximum amplitude, area and volume
        areas = self.pulses.keys()
        amplitudes = []
        volumes = []
        for area in areas:
            for cr in self.pulses[area]:
                value = abs(crh.get_value(cr))
                amplitudes.append(value)
                volumes.append(area * value)

        max_amplitude = max(amplitudes)
        max_volume = max(volumes)
        max_area = max(areas)

        self.add_trait('amplitude_threshold_min',
                       Range(value=0, low=0, high=max_amplitude)),
        self.add_trait('amplitude_threshold_max',
                       Range(value=max_amplitude, low=0, high=max_amplitude))

        max_area = max(pulses.keys())
        self.add_trait('area_threshold_min',
                       Range(value=0, low=0, high=max_area))
        self.add_trait('area_threshold_max',
                       Range(value=max_area, low=0, high=max_area))

        self.add_trait('volume_threshold_min',
                       Range(value=0, low=0, high=max_volume))
        self.add_trait('volume_threshold_max',
                       Range(value=max_volume, low=0, high=max_volume))

        self.result = self.image

    def _reconstruction_default(self):
        self.plot_data = ArrayPlotData(original=self.image,
                                       reconstruction=self.result)

        rows, cols = self.image.shape[:2]
        aspect = cols/float(rows)

        old = Plot(self.plot_data)
        old.img_plot('original', colormap=gray, origin='top left')
        old.title = 'Old'
        old.aspect_ratio = aspect

        self.new = Plot(self.plot_data)
        self.new.img_plot('reconstruction', colormap=gray, origin='top left')
        self.new.title = 'New'
        self.new.aspect_ratio = aspect

        container = HPlotContainer(bgcolor='none')
        container.add(old)
        container.add(self.new)

        return container

    @on_trait_change('amplitude_threshold_min, amplitude_threshold_max,'
                     'volume_threshold_min, volume_threshold_max,'
                     'area_threshold_min, area_threshold_max')
    def reconstruct(self):
        self.result.fill(0)

        # Reconstruct only from pulses inside the thresholds
        for area in self.pulses.keys():
            if area <= self.area_threshold_min or \
               area > self.area_threshold_max:
                continue

            for cr in self.pulses[area]:
                value = crh.get_value(cr)
                aval = abs(value)
                if aval <= self.amplitude_threshold_min or \
                   aval > self.amplitude_threshold_max:
                    continue

                volume = aval * area
                if volume <= self.volume_threshold_min or \
                   volume > self.volume_threshold_max:
                    continue

                crh.set_array(self.result, cr, value, 'add')

        self.plot_data.set_data('reconstruction', self.result)
        self.new.request_redraw()

image = load_image()
pulses = lulu.decompose(image)

viewer = Viewer(pulses=pulses, image=image)
viewer.configure_traits()
