from __future__ import print_function

from enable.api import Component, ComponentEditor
from traits.api import HasTraits, Instance, Array, Int, Range, \
                       on_trait_change, Dict, Bool, File, Button
from traitsui.api import Item, View, Group, HGroup, RangeEditor
from chaco.api import Plot, ArrayPlotData, PlotLabel, \
                      HPlotContainer, gray

from traitsui.api import DefaultOverride

import numpy as np

from demo import load_image

import lulu
import lulu.connected_region_handler as crh

class BaseViewer(HasTraits):
    reconstruction = Instance(Component)
    image = Array
    result = Array
    save_file = File(exists=False, auto_set=False, enter_set=True)
    save_button = Button('Save Result as .npy')

    def __init__(self, **kwargs):
        HasTraits.__init__(self, **kwargs)

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

    def update_plot(self):
        self.plot_data.set_data('reconstruction', self.result)
        self.new.request_redraw()

    def _save_button_changed(self):
        try:
            np.save(self.save_file, self.result)
        except IOError as e:
            message('Could not save file: %s' % str(e))

        try:
            f = open(self.save_file + '.txt', 'w')
            f.write(str(self))
            f.close()
        except IOError:
            message('Could not save file: %s' % str(e))



class Viewer(BaseViewer):
    pulses = Dict
    pulses_used = Int

    lifetimes = Array
    pulses_used = Int
    absolute_sum = Bool(False)
    amplitudes_one = Bool(False)
    replace = Bool(False)
    subtract = Bool(False)

    # Thresholds are defined in __init__

    def default_traits_view(self):
        no_endlabel = DefaultOverride(low_label='', high_label='',
                                      mode='logslider')
        no_endlabel_linear = DefaultOverride(low_label='', high_label='',
                                             mode='slider')

        return View(Group(Item('reconstruction', editor=ComponentEditor()),
                          show_labels=False,
                          show_left=False),
                    HGroup(Item('pulses_used', style='readonly'),
                           Item('absolute_sum'),
                           Item('amplitudes_one'),
                           Item('replace'),
                           Item('subtract')),
                    Item('amplitude_threshold_min', editor=no_endlabel,
                         label='Minimum absolute amplitude'),
                    Item('amplitude_threshold_max', editor=no_endlabel),
                    Item('area_threshold_min', editor=no_endlabel),
                    Item('area_threshold_max', editor=no_endlabel),
                    Item('volume_threshold_min', editor=no_endlabel),
                    Item('volume_threshold_max', editor=no_endlabel),
                    Item('circularity_min'),
                    Item('circularity_max'),
                    Item('lifetime_min', editor=no_endlabel_linear),
                    Item('lifetime_max', editor=no_endlabel_linear),
                    Item('output_threshold', editor=no_endlabel_linear),

                    HGroup(Item('save_file', show_label=False),
                           Item('save_button', show_label=False)
                          ),

                    width=800, height=600,
                    resizable=True,
                    title='DPT 2D reconstruction')

    def __init__(self, **kwargs):
        BaseViewer.__init__(self, **kwargs)

        # Calculate maximum amplitude, area and volume
        areas = self.pulses.keys()
        amplitudes = []
        volumes = []
        for area in areas:
            self.pulses_used += len(self.pulses[area])
            for cr in self.pulses[area]:
                value = abs(crh.get_value(cr))
                amplitudes.append(value)
                volumes.append(area * value)

        max_amplitude = max(amplitudes)
        max_volume = max(volumes)
        max_area = max(areas)

        # Calculate lifetimes
        life_starts = np.zeros_like(self.image)
        for area in reversed(sorted(self.pulses.keys())):
            for cr in self.pulses[area]:
                crh.set_array(life_starts, cr, area)

        life_ends = np.zeros_like(self.image)
        for area in sorted(self.pulses.keys()):
            for cr in self.pulses[area]:
                crh.set_array(life_ends, cr, area)

        self.lifetimes = life_ends - life_starts

        self.add_trait('amplitude_threshold_min',
                       Range(value=1, low=1, high=max_amplitude)),
        self.add_trait('amplitude_threshold_max',
                       Range(value=max_amplitude, low=1, high=max_amplitude))

        max_area = max(self.pulses.keys())
        self.add_trait('area_threshold_min',
                       Range(value=1, low=1, high=max_area))
        self.add_trait('area_threshold_max',
                       Range(value=max_area, low=1, high=max_area))

        self.add_trait('volume_threshold_min',
                       Range(value=1, low=1, high=max_volume))
        self.add_trait('volume_threshold_max',
                       Range(value=max_volume, low=1, high=max_volume))

        self.add_trait('circularity_min',
                       Range(value=0, low=0, high=1.0))
        self.add_trait('circularity_max',
                       Range(value=1.0, low=0, high=1.0))

        self.add_trait('lifetime_min',
                       Range(value=int(self.lifetimes.min()),
                             low=int(self.lifetimes.min()),
                             high=int(self.lifetimes.max())))
        self.add_trait('lifetime_max',
                       Range(value=int(self.lifetimes.max()),
                             low=int(self.lifetimes.min()),
                             high=int(self.lifetimes.max())))

        self.add_trait('output_threshold',
                       Range(value=0,
                             low=0,
                             high=255))

        self.result = self.image.copy()

    @on_trait_change('amplitude_threshold_min, amplitude_threshold_max,'
                     'volume_threshold_min, volume_threshold_max,'
                     'area_threshold_min, area_threshold_max,'
                     'circularity_min, circularity_max,'
                     'absolute_sum, amplitudes_one, lifetime_min,'
                     'lifetime_max, replace, subtract,'
                     'output_threshold')
    def reconstruct(self):
        self.result.fill(0)
        pulses = 0

        # Reconstruct only from pulses inside the thresholds
        for area in sorted(self.pulses.keys()):
            if area < self.area_threshold_min or \
               area > self.area_threshold_max:
                continue

            for cr in self.pulses[area]:
                value = crh.get_value(cr)
                aval = abs(value)
                if aval < self.amplitude_threshold_min or \
                   aval > self.amplitude_threshold_max:
                    continue

                volume = aval * area
                if volume < self.volume_threshold_min or \
                   volume > self.volume_threshold_max:
                    continue

                ## # See:
                ## #
                ## # Measuring rectangularity by Paul L. Rosin
                ## # Machine Vision and Applications, Vol 11, No 4, December 1999
                ## # http://www.springerlink.com/content/xb9klcax8ytnwth1/
                ## #
                ## # for more information on computing rectangularity.
                ## #
                ## r0, c0, r1, c1 = crh.bounding_box(cr)
                ## if c0 == c1 or r0 == r1:
                ##     rectangularity = 1
                ## else:
                ##     rectangularity = area / float((c1 - c0 + 1) * (r1 - r0 + 1))

                ## if rectangularity < self.rectangularity_min or \
                ##    rectangularity > self.rectangularity_max:
                ##     continue

                r0, c0, r1, c1 = crh.bounding_box(cr)
                if (c0 == c1) and (r0 == r1):
                    circularity = 1
                else:
                    max_dim = max(abs(r0 - r1), abs(c0 - c1))
                    circularity = crh.nnz(cr) / \
                                  (np.pi / 4 * (max_dim + 2)**2)
                    # We add 2 to max_dim to allow for pixel overlap
                    # with circumscribing circle
                if circularity < self.circularity_min or \
                   circularity > self.circularity_max:
                    continue

                if self.absolute_sum:
                    value = aval

                if self.amplitudes_one:
                    value = 1

                if self.replace:
                    crh.set_array(self.result, cr, value)
                else:
                    crh.set_array(self.result, cr, value, 'add')

                pulses += 1

        mask = (self.lifetimes > self.lifetime_max) | \
               (self.lifetimes < self.lifetime_min)
        self.result[mask] = 0

        if self.subtract:
            self.result = np.abs(self.image - self.result)

        if self.output_threshold != 0:
            self.result[self.result <= self.output_threshold] = 0

        self.pulses_used = pulses

        self.update_plot()

    def __str__(self):
        def pptrait(tname):
            return ' '.join(tname.split('_')).capitalize()

        def bracket(tname):
            pname = pptrait(tname)
            tmin = getattr(self, tname + '_min')
            tmax = getattr(self, tname + '_max')
            return ('%s: [%s, %s]' % (pname, tmin, tmax))

        out = '\n'.join(bracket(t) for t in
                        ('amplitude_threshold', 'area_threshold',
                         'volume_threshold', 'circularity', 'lifetime'))

        for tname in ('absolute_sum', 'amplitudes_one', 'replace', 'subtract',
                      'output_threshold', 'pulses_used'):
            out += ('\n%s: %s' % (pptrait(tname), getattr(self, tname)))

        return out


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and '-UL' in sys.argv:
        operator = 'UL'
        sys.argv.remove('-UL')
    else:
        operator = 'LU'

    image = load_image()

    print("Decomposing using the %s operator." % operator)
    if operator == 'LU':
        print("Use the '-UL' flag to switch to UL.")

    print()
    pulses = lulu.decompose(image, operator=operator)

    viewer = Viewer(pulses=pulses, image=image)
    viewer.configure_traits()
