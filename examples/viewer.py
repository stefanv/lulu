from enthought.enable.api import Component, ComponentEditor
from enthought.traits.api import HasTraits, Instance, Array, Int, Range
from enthought.traits.ui.api import Item, View, Group, RangeEditor
from enthought.chaco.api import Plot, ArrayPlotData, PlotLabel, \
                                HPlotContainer


import numpy as np

class Viewer(HasTraits):
    reconstruction = Instance(Component)

    result = Array

    # Thresholds are defined in __init__

    traits_view = View(Group(Item('reconstruction', editor=ComponentEditor()),
                             show_labels=False,
                             show_left=False),
                       Item('amplitude_threshold_min'),
                       Item('amplitude_threshold_max'),
                       Item('area_threshold_min'),
                       Item('area_threshold_max'),
                       Item('volume_threshold_min'),
                       Item('volume_threshold_max'),

                       resizable=True,
                       title='DPT 2D reconstruction')

    def __init__(self, impulses):
        HasTraits.__init__(self)

        def _slider(low, high):
            return RangeEditor(low=low, high=high,
                               high_label='', low_label=' ',
                               mode='slider')


        self.add_trait('amplitude_threshold_min',
                       Range(value=0, low=0, high=10,
                             editor=_slider(0, 10)))
        self.add_trait('amplitude_threshold_max',
                       Range(value=10, low=0, high=10,
                             editor=_slider(0, 10)))

        max_area = max(impulses.keys())
        self.add_trait('area_threshold_min',
                       Range(value=0, low=0, high=max_area,
                             editor=_slider(0, max_area)))
        self.add_trait('area_threshold_max',
                       Range(value=max_area, low=0, high=max_area,
                             editor=_slider(0, max_area)))

        self.add_trait('volume_threshold_min',
                       Range(value=0, low=0, high=1500,
                             editor=_slider(0, 1500)))
        self.add_trait('volume_threshold_max',
                       Range(value=1500, low=0, high=1500,
                             editor=_slider(0, 1500)))

    def _reconstruction_default(self):
        index = np.arange(5)
        data_series = index**2

        plot_data = ArrayPlotData(index=index)
        plot_data.set_data('data_series', data_series)

        original = Plot(plot_data)
        original.plot(('index', 'data_series'))
        original.title = 'Old'

        reconstruction = Plot(plot_data)
        reconstruction.plot(('index', 'data_series'))
        reconstruction.title = 'New'

        container = HPlotContainer(bgcolor='none')
        container.add(original)
        container.add(reconstruction)

        return container

viewer = Viewer(impulses={0:[], 10:[], 100:[]})
viewer.configure_traits()
