import pandas as pd 
import numpy as np 
import pyaudio
import wave
import sys
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import pylab as plt
import os

def open_file(file):
    '''
    Input: String with the file location of a sound

    Output: A list of values pertaining to the sound file

    Reads in Wav files and extracts the numerical data to be used
    by within the file.

    '''

    fs, data = wavfile.read(file)
    trans_data =[(ele/2**8.)*2-1 for ele in data]

    return trans_data

def get_features(file, rank_type='mels'):
    '''
    Input: file: String with the file location of a sound,
           sound_type: String with the label for the data,
           rank_type: String 'mels' or 'log' to describe bucket type



    Output: Dictionary containing all the bucket values and label for
    for the data

    Takes a file name and converts it to its feature row
    
    '''
    #Start and finish for every bucket in mel's distribution
    mels = [1,200,400,630,920,1270,1720,2320,3200,6400]
    
    sound = open_file(file):
    sound_fft = fft(sound)
    
    bucket_scores={}

    for i,x in enumerate(rank_type):
        print i
        if 1 == 0:
            continue
        else:
            bucket_scores[rank_type['bucket'+str(i)]] = sum(sound_fft[x-1:x])

    sound_type = file.split('/')[-2]
    bucket_scores['type'] = sound_type

    return bucket_scores

def build_feature_matrix(data_location):
    sounds = os.listdir(data_location)

    feature_list = []
    for sound in sounds:
        files = os.listdir(data_location+ '/' + sound)
        for a_file in files:
            feature_list.append(get_features(data_location+ '/' + sound + '/' +a_file))

    return feature_list



