# Major library imports
import pyaudio
from numpy import zeros, linspace, short, fromstring, transpose, array
from scipy import fft
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import cPickle as pickle

# Enthought library imports
from enable.api import Window, Component, ComponentEditor
from traits.api import HasTraits, Instance, List, Enum
from traitsui.api import Item, Group, View, Handler
from pyface.timer.api import Timer
#from enthought.traits.ui.menu import OKButton, CancelButton

# Chaco imports
from chaco.api import (Plot, ArrayPlotData, HPlotContainer, VPlotContainer,
    AbstractMapper, LinePlot, LinearMapper, DataRange1D, BarPlot)

#Atributes for the stream    
NUM_SAMPLES = 1024
SAMPLING_RATE = 22050 #11025
SPECTROGRAM_LENGTH = 50

#import model
forest = pickle.load(open('/Users/CPinkston/Documents/Zipfian/FreedomOfSpeech/data/mono_model.p'))

# Attributes to use for the plot view.
size = (900,850)
title = "Speech Visualizer"

# Visual guide for sound
background_dict = {'AA':[ 35.81397578,  45.01631462,  56.12467144,  12.1484378 ,
          3.06433121,   1.91775815,   3.08735659,   5.32401947,
          5.88404492,   7.95661794,   1.88881055,   1.42798042,
          3.96340981,   2.68320701,   1.9224381 ,   1.11476687],
       'AE':[ 27.4362838 ,  27.10570391,  39.44961208,  12.29018975,
         17.05977796,   7.38096405,  11.72786887,   5.76044961,
          6.53821816,   4.01565196,   1.38825463,   2.09123181,
          2.77621646,   1.32513547,   2.10078532,   1.66806562],
       'AH':[ 37.75816345,  55.48377506,  27.80481956,  24.44436866,
          3.1186638 ,   1.66242775,   5.13277218,   6.03904341,
          9.05814786,   4.44443824,   1.02058099,   1.65270508,
          2.93098263,   4.06752972,   1.6135137 ,   0.81619345],
       'AO':[ 21.47497231,  38.28943443,  50.62903415,   8.35393119,
          3.07678339,   2.05409526,   2.37562784,   2.00558598,
          3.01447916,   4.26127112,   1.80530714,   3.73105018,
          4.0233738 ,   2.07571379,   2.07032168,   1.29067984],
       'EH':[ 26.22553365,  22.12785448,  12.18583196,   4.79216754,
          9.64753079,   6.19215511,   7.10419455,   4.58124136,
          2.65370963,   2.79506782,   0.82444456,   1.82331211,
          1.80034007,   1.76603673,   1.79746801,   0.58000965],
       'ER':[ 55.3946186 ,  56.88201025,  21.1990828 ,  20.20514299,
          4.89875334,   1.44183263,   1.45997099,   1.14090613,
          1.45707253,   0.98617129,   1.56760651,   2.98495273,
          1.36142933,   1.09301032,   1.55692499,   1.14341959],
       'EY':[ 39.19160161,  34.85463803,   4.64036407,   2.60255225,
          2.95812851,   6.98613796,   4.91738817,   4.13260074,
          1.85325952,   1.09616616,   1.35829042,   1.11305425,
          1.5054683 ,   1.3815847 ,   1.64839403,   1.14386384],
       'IH':[ 48.58006533,  54.00788813,   7.872923  ,   3.49289675,
          4.49289906,  10.40966844,   7.84612386,   7.95802725,
          5.29840227,   1.94301648,   1.99454648,   1.51040584,
          2.0216913 ,   3.02829396,   3.92954506,   1.90158391],
       'IY':[ 46.87647265,   9.43369867,   1.80606519,   1.05745142,
          0.86969495,   1.49165165,   3.07262719,   2.78754269,
          2.94247314,   1.10598322,   1.38521549,   1.90596473,
          1.68691551,   3.4546833 ,   1.49868147,   0.90018443],
       'OW':[ 27.38729711,  42.73110212,  19.68563542,   3.67982251,
          2.04154194,   1.52837558,   2.75657802,   1.3685308 ,
          2.38274303,   1.75837408,   1.22286706,   1.76399616,
          1.77150212,   1.86287625,   1.39946777,   1.11275704],
       'UH':[ 29.62068641,  33.75635882,   8.2471728 ,  10.92877852,
          2.18208943,   1.00674882,   3.37916534,   3.20349283,
          2.04546681,   5.33045881,   2.23010159,   1.43233783,
          3.53001245,   1.31270093,   1.4417948 ,   0.62741355],
       'UW':[ 29.84144362,  18.36840188,   6.89481727,   1.60053398,
          0.98458783,   0.75937742,   0.83373466,   0.63936673,
          0.67474447,   0.57706845,   0.62018357,   0.58136584,
          0.74068797,   0.61835592,   0.84554567,   0.66147225]}

def get_audio_data():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                     input=True, frames_per_buffer=NUM_SAMPLES)
    audio_data  = fromstring(stream.read(NUM_SAMPLES), dtype=short)
    stream.close()
    normalized_data = audio_data / 32768.0
    sound_fft = abs(fft(normalized_data))[:NUM_SAMPLES/2]
    
    bucket_scores=[]
    #mels = array([0,200,400,630,920,1270,1720,2320,3200])/6.25
    mels = array([0,100,200,300,400,515,630,780,920,1095,1270,1495,1720,2020,2320,2769,3200])/6.25
    
    
    plot_type = popup.plot_type
    
    for i,x in enumerate(mels):
        if i == 0:
            continue
        else:
            bucket_scores.append(sum(sound_fft[mels[i-1]:x]))
    classification = ['Spectrum']
    if bucket_scores[0] > 5:
        classification =  forest.predict(bucket_scores)
        
    return (array(bucket_scores), normalized_data, classification, plot_type)

def _create_plot_component(obj):
    # Setup the spectrum plot
    
    frequencies = linspace(0.0, 16, num=16) ##
    obj.spectrum_data = ArrayPlotData(frequency=frequencies)
    empty_amplitude = zeros(16)##
    obj.spectrum_data.set_data('amplitude', empty_amplitude)
    background = array([50,0,0,0,0,0,0,0,50,0,0,0,0,0,0,0])*20
    obj.spectrum_data.set_data('background', background)

    obj.spectrum_plot = Plot(obj.spectrum_data)
    
    obj.spectrum_plot.plot(("frequency", "background"), value_scale="linear", type='bar', bar_width=0.8, color='auto')
    obj.spectrum_plot.plot(("frequency", "amplitude"), value_scale="linear", name="Spectrum", type='bar', bar_width=0.8, color='auto')#[0]
    obj.spectrum_plot.padding = 50
    obj.spectrum_plot.title = "Spectrum"
    spec_range = obj.spectrum_plot.plots.values()[0][0].value_mapper.range
    spec_range.low = 0.0
    spec_range.high = 50.0
    obj.spectrum_plot.index_axis.title = 'Frequency (hz)'
    obj.spectrum_plot.value_axis.title = 'Amplitude'
    
    container = HPlotContainer()
    container.add(obj.spectrum_plot)

    c2 = VPlotContainer()
    c2.add(container)

    return c2

class TimerController(HasTraits):

    def onTimer(self, *args):
        spectrum, time, classification, plot_type = get_audio_data()
        self.spectrum_data.set_data('amplitude', spectrum)
        self.spectrum_plot.title = classification[0]
        self.spectrum_data.set_data('background',background_dict[plot_type])
        #spec_data = self.spectrogram_plot.values[1:] + [spectrum]
        #self.spectrogram_plot.values = spec_data
        self.spectrum_plot.request_redraw()


class DemoHandler(Handler):

    def closed(self, info, is_ok):
        """ Handles a dialog-based user interface being closed by the user.
        Overridden here to stop the timer once the window is destroyed.
        """

        info.object.timer.Stop()
    

class Demo(HasTraits):

    plot = Instance(Component)

    controller = Instance(TimerController, ())

    timer = Instance(Timer)
    
    plot_type = Enum("AA", "AE","AH", "AO", "EH", "ER", "EY", "IH", "IY", "OW", 
    "UH", "UW")

    traits_view = View(
                    Group(
                        Item('plot', editor=ComponentEditor(size=size),
                             show_label=False),
                        orientation = "vertical"),
                        Item(name='plot_type'),
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
        self.timer = Timer(10, self.controller.onTimer)
        return super(Demo, self).configure_traits(*args, **kws)

popup = Demo()

if __name__ == "__main__":
    popup.configure_traits()