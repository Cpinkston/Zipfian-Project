# Major library imports
import pyaudio
from numpy import zeros, linspace, short, fromstring, transpose, array
from scipy import fft

# Enthought library imports
from enable.api import Window, Component, ComponentEditor
from traits.api import HasTraits, Instance, List
from traitsui.api import Item, Group, View, Handler
from pyface.timer.api import Timer

# Chaco imports
from chaco.api import (Plot, ArrayPlotData, HPlotContainer, VPlotContainer,
    AbstractMapper, LinePlot, LinearMapper, DataRange1D, BarPlot)

#Atributes for the stream    
NUM_SAMPLES = 1024
SAMPLING_RATE = 11025
SPECTROGRAM_LENGTH = 50


def get_audio_data():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                     input=True, frames_per_buffer=NUM_SAMPLES)
    audio_data  = fromstring(stream.read(NUM_SAMPLES), dtype=short)
    stream.close()
    normalized_data = audio_data / 32768.0
    sound_fft = abs(fft(normalized_data))[:NUM_SAMPLES/2]
    
    bucket_scores=[]
    mels = array([1,200,400,630,920,1270,1720,2320,3200,6400])/4

    for i,x in enumerate(mels):
        if i == 0:
            continue
        else:
            bucket_scores.append(sum(sound_fft[mels[i-1]:x]))
    
    return (array(bucket_scores), normalized_data)

def _create_plot_component(obj):
    # Setup the spectrum plot
    
    frequencies = linspace(0.0, 9, num=9)
    obj.spectrum_data = ArrayPlotData(frequency=frequencies)
    empty_amplitude = zeros(9)
    obj.spectrum_data.set_data('amplitude', empty_amplitude)
    background = array([3.5,.5,.25,.5,1,.5,.1,0,0])*20
    obj.spectrum_data.set_data('background', background)

    obj.spectrum_plot = Plot(obj.spectrum_data)
    
    obj.spectrum_plot.plot(("frequency", "background"), type='bar', bar_width=0.8, color='auto')
    obj.spectrum_plot.plot(("frequency", "amplitude"), name="Spectrum", type='bar', bar_width=0.8, color='auto')#[0]
    obj.spectrum_plot.padding = 50
    obj.spectrum_plot.title = "Spectrum"
    spec_range = obj.spectrum_plot.plots.values()[0][0].value_mapper.range
    spec_range.low = 0.0
    spec_range.high = 100.0
    obj.spectrum_plot.index_axis.title = 'Frequency (hz)'
    obj.spectrum_plot.value_axis.title = 'Amplitude'
    
    container = HPlotContainer()
    container.add(obj.spectrum_plot)

    c2 = VPlotContainer()
    c2.add(container)

    return c2

class TimerController(HasTraits):

    def onTimer(self, *args):
        spectrum, time = get_audio_data()
        self.spectrum_data.set_data('amplitude', spectrum)
        #spec_data = self.spectrogram_plot.values[1:] + [spectrum]
        #self.spectrogram_plot.values = spec_data
        self.spectrum_plot.request_redraw()
        return


class DemoHandler(Handler):

    def closed(self, info, is_ok):
        """ Handles a dialog-based user interface being closed by the user.
        Overridden here to stop the timer once the window is destroyed.
        """

        info.object.timer.Stop()
        return

# Attributes to use for the plot view.
size = (900,850)
title = "Audio Spectrum Waterfall"

class Demo(HasTraits):

    plot = Instance(Component)

    controller = Instance(TimerController, ())

    timer = Instance(Timer)

    traits_view = View(
                    Group(
                        Item('plot', editor=ComponentEditor(size=size),
                             show_label=False),
                        orientation = "vertical"),
                    resizable=True, title=title,
                    width=size[0], height=size[1]+25,
                    handler=DemoHandler
                    )

    def __init__(self, **traits):
        super(Demo, self).__init__(**traits)
        self.plot = _create_plot_component(self.controller)

    def edit_traits(self, *args, **kws):
        # Start up the timer! We should do this only when the demo actually
        # starts and not when the demo object is created.
        self.timer = Timer(20, self.controller.onTimer)
        return super(Demo, self).edit_traits(*args, **kws)

    def configure_traits(self, *args, **kws):
        # Start up the timer! We should do this only when the demo actually
        # starts and not when the demo object is created.
        self.timer = Timer(20, self.controller.onTimer)
        return super(Demo, self).configure_traits(*args, **kws)

popup = Demo()

if __name__ == "__main__":
    popup.configure_traits()