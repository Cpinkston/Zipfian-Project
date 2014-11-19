# Major library imports
import pyaudio
from numpy import zeros, linspace, short, fromstring, transpose, array, ones
from scipy import fft
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import cPickle as pickle

# Enthought library imports
from enable.api import Window, Component, ComponentEditor
from traits.api import HasTraits, Instance, List, Enum, Str
from traitsui.api import Item, Group, View, Handler
from pyface.timer.api import Timer
from enthought.traits.ui.menu import OKButton, CancelButton

# Chaco imports
from chaco.api import (Plot, ArrayPlotData, HPlotContainer, VPlotContainer,
    AbstractMapper, LinePlot, LinearMapper, DataRange1D, BarPlot)

#Atributes for the stream    
NUM_SAMPLES = 1024
SAMPLING_RATE = 22050 #11025
SPECTROGRAM_LENGTH = 50

# Attributes to use for the plot view.
size = (900,850)
title = "Speech Visualizer"

#import models
mono = pickle.load(open('/Users/CPinkston/Documents/Zipfian/FreedomOfSpeech/data/mono_model.p'))
stops = pickle.load(open('/Users/CPinkston/Documents/Zipfian/FreedomOfSpeech/data/stops_model.p'))
dip = pickle.load(open('/Users/CPinkston/Documents/Zipfian/FreedomOfSpeech/data/dip_model.p'))
other = pickle.load(open('/Users/CPinkston/Documents/Zipfian/FreedomOfSpeech/data/other_model.p'))
frics = pickle.load(open('/Users/CPinkston/Documents/Zipfian/FreedomOfSpeech/data/fricatives_model.p'))

forest = mono

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
          
word_list = pickle.load(open('/Users/CPinkston/Documents/Zipfian/FreedomOfSpeech/data/word_dict.p'))
            
sound_fft_bank = pickle.load(open('/Users/CPinkston/Documents/Zipfian/FreedomOfSpeech/data/spectro_dict.p'))

def get_audio_data():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                     input=True, frames_per_buffer=NUM_SAMPLES)
    audio_data  = fromstring(stream.read(NUM_SAMPLES), dtype=short)
    stream.close()
    normalized_data = audio_data / 32768.0
    sound_fft = abs(fft(normalized_data))[:NUM_SAMPLES/2]
    
    bucket_scores=[]
    mels = array([0,100,200,300,400,515,630,780,920,1095,1270,1495,1720,\
    2020,2320,2769,3200])/6.25
    
    
    plot_type = popup.plot_type
    word = popup.word.upper()
    word_shape = [zeros(NUM_SAMPLES/4) for i in xrange(SPECTROGRAM_LENGTH)]
    pronunciation = ''
     
    
    if word in word_list:
        for i, sound in enumerate(word_list[word]):
            word_shape[45-(i*3)] = sound_fft_bank[word_list[word][-(i+1)]]*4
            word_shape[45-((i*3)+1)] = sound_fft_bank[word_list[word][-(i+1)]]*4
            word_shape[45-((i*3)+2)] = sound_fft_bank[word_list[word][-(i+1)]]*4
            pronunciation = pronunciation + sound + " "
        pronunciation = [pronunciation]
    
    for i,x in enumerate(mels):
        if i == 0:
            continue
        else:
            bucket_scores.append(sum(sound_fft[mels[i-1]:x]))
    
    if popup.model_type == 'Monothongs':
        forest = mono 
    elif popup.model_type == 'Fricatives': 
        forest = frics  
    elif popup.model_type == 'Diphthongs':
        forest = dip
    elif popup.model_type == 'Stops':
        forest = stops
    else:
        forest = other  
    
    if bucket_scores[0] > 5:
        classification =  forest.predict(bucket_scores)
        if pronunciation == '':
            pronunciation = classification
        else:
            pronunciation = pronunciation[0] + '--' + classification    
        
    if pronunciation == '':
        pronunciation = ['Word Visualizer']
        
    return (array(bucket_scores), normalized_data, pronunciation, plot_type,\
    sound_fft[:len(sound_fft)/2], word_shape)

def _create_plot_component(obj):
    # Setup the spectrum plot
    
    frequencies = linspace(0.0, 16, num=16) ##
    obj.spectrum_data = ArrayPlotData(frequency=frequencies)
    empty_amplitude = zeros(16)##
    obj.spectrum_data.set_data('amplitude', empty_amplitude)
    background = array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])*20
    obj.spectrum_data.set_data('background', background)

    obj.spectrum_plot = Plot(obj.spectrum_data)
    
    obj.spectrum_plot.plot(("frequency", "background"), value_scale="linear",\
    type='bar', bar_width=0.8, color='auto')
    obj.spectrum_plot.plot(("frequency", "amplitude"), value_scale="linear",\
    name="Spectrum", type='bar', bar_width=0.8, color='auto')#[0]
    obj.spectrum_plot.padding = 50
    obj.spectrum_plot.title = "Spectrum"
    spec_range = obj.spectrum_plot.plots.values()[0][0].value_mapper.range
    spec_range.low = 0.0
    spec_range.high = 75.0
    obj.spectrum_plot.index_axis.title = 'Frequency (hz)'
    obj.spectrum_plot.value_axis.title = 'Amplitude'
    
    # Word Plots
    # Setup the unseen spectrum plot
    dum_frequencies = linspace(0.0, float(SAMPLING_RATE)/4, num=NUM_SAMPLES/4)
    obj.dum_spectrum_data = ArrayPlotData(frequency=dum_frequencies)
    dum_empty_amplitude = zeros(NUM_SAMPLES/4)
    obj.dum_spectrum_data.set_data('amplitude', dum_empty_amplitude)

    obj.dum_spectrum_plot = Plot(obj.dum_spectrum_data)
    spec_renderer = obj.dum_spectrum_plot.plot(("frequency", "amplitude"),
    name="Spectrum",color="red")[0]
    
    values = [zeros(NUM_SAMPLES/4) for i in xrange(SPECTROGRAM_LENGTH)]
    
    p = WaterfallRenderer(index = spec_renderer.index, values = values,
            index_mapper = LinearMapper(range = obj.dum_spectrum_plot.\
            index_mapper.range),
            value_mapper = LinearMapper(range = DataRange1D(low=0,\
            high=SPECTROGRAM_LENGTH)),
            y2_mapper = LinearMapper(low_pos=0, high_pos=8,
                            range=DataRange1D(low=0, high=15)),
            )
    spectrogram_plot = p
    obj.spectrogram_plot = p
    dummy = Plot()
    dummy.title = "Current Spectral Envelope" 
    dummy.padding = 50
    dummy.index_axis.mapper.range = p.index_mapper.range
    dummy.index_axis.title = "Frequency (hz)"
    dummy.add(p)
    
    values2 = [zeros(NUM_SAMPLES/4) for i in xrange(SPECTROGRAM_LENGTH)]
    p2 = WaterfallRenderer(index = spec_renderer.index, values = values2,
            index_mapper = LinearMapper(range = obj.dum_spectrum_plot.\
            index_mapper.range),
            value_mapper = LinearMapper(range = DataRange1D(low=0,\
            high=SPECTROGRAM_LENGTH)),
            y2_mapper = LinearMapper(low_pos=0, high_pos=8,
                            range=DataRange1D(low=0, high=15)),
            )
    spectrogram_plot2 = p2
    obj.spectrogram_plot2 = p2
    dummy2 = Plot()
    dummy2.title = "Target Spectral Envelope" 
    dummy2.padding = 50
    dummy2.index_axis.mapper.range = p.index_mapper.range
    dummy2.index_axis.title = "Frequency (hz)"
    dummy2.add(p2)
    
    
    container = HPlotContainer()
    container.add(dummy)
    container.add(dummy2)

    c2 = VPlotContainer()
    c2.add(container)
    c2.add(obj.spectrum_plot)

    return c2

class TimerController(HasTraits):

    def onTimer(self, *args):
        spectrum, time, classification, plot_type, sound_fft, word_shape =\
        get_audio_data()
        self.spectrum_data.set_data('amplitude', spectrum)
        self.spectrum_plot.title = classification[0]
        self.spectrum_data.set_data('background',background_dict[plot_type])
        spec_data = self.spectrogram_plot.values[1:] + [sound_fft]
        self.spectrogram_plot.values = spec_data
        self.spectrogram_plot2.values = word_shape
        self.spectrum_plot.request_redraw()



class DemoHandler(Handler):

    def closed(self, info, is_ok):
        """ Handles a dialog-based user interface being closed by the user.
        Overridden here to stop the timer once the window is destroyed.
        """

        info.object.timer.Stop()
    

class WaterfallRenderer(LinePlot):

    # numpy arrays of the same length
    values = List(args=[])

    # Maps each array in values into a contrained, short screen space
    y2_mapper = Instance(AbstractMapper)

    _cached_data_pts = List()
    _cached_screen_pts = List()

    def _gather_points(self):
        if not self._cache_valid:
            if not self.index or len(self.values) == 0:
                return

            index = self.index.get_data()
            values = self.values

            numindex = len(index)
            if numindex == 0 or all(len(v)==0 for v in values) or\
            all(numindex != len(v) for v in values):
                self._cached_data_pts = []
                self._cache_valid = True

            self._cached_data_pts = \
            [transpose(array((index, v))) for v in values]
            self._cache_value = True
        return

    def get_screen_points(self):
        self._gather_points()
        return [self.map_screen(pts, i) for i, pts in enumerate\
        (self._cached_data_pts)]

    def map_screen(self, data_array, data_offset=None):
        """ data_offset, if provided, is a float that will be mapped
        into screen space using self.value_mapper and then added to
        mapping data_array with y2_mapper.  If data_offset is not
        provided, then y2_mapper is used.
        """
        if len(data_array) == 0:
            return []
        x_ary, y_ary = transpose(data_array)
        sx = self.index_mapper.map_screen(x_ary)
        if data_offset is not None:
            dy = self.value_mapper.map_screen(data_offset)
            sy = self.y2_mapper.map_screen(y_ary) + dy
        else:
            sy = self.value_mapper.map_screen(y_ary)

        if self.orientation == "h":
            return transpose(array((sx, sy)))
        else:
            return transpose(array((sy, sx)))


class Demo(HasTraits):

    plot = Instance(Component)
    controller = Instance(TimerController, ())
    timer = Instance(Timer)
    
    plot_type = Enum("AA", "AE","AH", "AO", "EH", "ER", "EY", "IH",
    "IY", "OW", "UH", "UW")    
    model_type = Enum('Monothongs','Fricatives', 'Diphthongs', 'Stops',
    'Others')     
    word = Str("hello(A)")

    traits_view = View(
                    Group(
                        Item('plot', editor=ComponentEditor(size=size),
                             show_label=False),
                        orientation = "vertical"),
                        Item(name='plot_type'),
                        Item(name='model_type'),
                        Item(name='word'),
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