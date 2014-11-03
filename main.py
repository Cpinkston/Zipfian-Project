import pandas as pd 
import numpy as np 
import pyaudio
import wave
import sys
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import pylab as plt

def open_file(file):
    '''
    Input: String with the file location of a sound

    Output: a list of values pertaining to the sound file

    Reads in Wav files and extracts the numerical data to be used
    by within the file.

    '''

    fs, data = wavfile.read(file)
    trans_data =[(ele/2**8.)*2-1 for ele in data]

    return trans_data