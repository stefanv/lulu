from __future__ import print_function

import numpy as np
from viewer import Viewer

from traits.api import Array
from chaco.api import Plot, ArrayPlotData, HPlotContainer, gray

import lulu

class Viewer1D(Viewer):
    image = Array
    result = Array

    def _reconstruction_default(self):
        rows, cols = self.image.shape[:2]
        self.plot_data = ArrayPlotData(original=self.image[0],
                                       reconstruction=self.result[0])

        aspect = cols/float(rows)

        old = Plot(self.plot_data)
        old.plot('original', )
        old.title = 'Old'

        self.new = Plot(self.plot_data)
        self.new.plot('reconstruction')
        self.new.title = 'New'

        container = HPlotContainer(bgcolor='none')
        container.add(old)
        container.add(self.new)

        return container

    def update_plot(self):
        self.plot_data.set_data('reconstruction', self.result[0])
        self.new.request_redraw()



if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 2 and '-UL' in sys.argv:
        operator = 'UL'
        sys.argv.remove('-UL')
    else:
        operator = 'LU'

    image = (np.sin(np.linspace(-np.pi, np.pi, 100))*100).astype(int)

    mask = np.random.random((100,)) > 0.8
    noise = np.random.random((100,))*100 - 50

    image += (mask * noise).astype(image.dtype)
    image = image.reshape((1,-1))

    print("Decomposing using the %s operator." % operator)
    if operator == 'LU':
        print("Use the '-UL' flag to switch to UL.")

    print()
    pulses = lulu.decompose(image, operator=operator)

    viewer = Viewer1D(pulses=pulses, image=image)
    viewer.configure_traits()
