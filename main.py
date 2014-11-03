import pandas as pd 
import numpy as np 
import pyaudio
import wave
import sys
from scipy.io import wavfile
from scipy.fftpack import fft, ifft
import pylab as plt
import os
from sklearn.ensemble import RandomForestClassifier

def open_file(file_name):
    '''
    Input: String with the file location of a sound

    Output: A list of values pertaining to the sound file

    Reads in Wav files and extracts the numerical data to be used
    by within the file.

    '''

    fs, data = wavfile.read(file_name)
    trans_data =[(ele/2**8.)*2-1 for ele in data]

    return trans_data

def get_features(file_name, rank_type='mels'):
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
    
    sound = open_file(file_name)
    sound_fft = fft(sound)
    sound_fft = abs(sound_fft[:(len(sound_fft)/2)])
    
    bucket_scores={}

    for i,x in enumerate(mels):
        if 1 == 0:
            continue
        else:
            name = 'bucket'+str(i)
            bucket_scores[name] = sum(sound_fft[x-1:x])

    sound_type = file_name.split('/')[-2]
    bucket_scores['type'] = sound_type

    return bucket_scores

def build_feature_matrix(data_location):
    '''
    Input: A string that is the location of a folder that contains folders of different sounds

    Output: A list of dictionaries that correspond to each sound within the specified folder

    A function to create a feature list of all the sounds
    '''

    sounds = os.listdir(data_location)

    feature_list = []
    for sound in sounds:
        if sound == '.DS_Store':
            continue
        files = os.listdir(data_location+ '/' + sound)
        for a_file in files:
            feature_list.append(get_features(data_location+ '/' + sound + '/' +a_file))

    return feature_list

def fit_model(folder):
    '''
    Input: A string that is the location of a folder that contains folders of different sounds
    
    Output: A fitted model to predict a sound

    This will take in test data to create a model to make predictions
    '''    

    sound_data = build_feature_matrix(folder)
    df = pd.DataFrame(sound_data)
    y = df.pop('type')

    forest = RandomForestClassifier(n_estimators=20)
    forest.fit(df.values, y.values)
    return forest



if __name__ == "__main__":
    model = fit_model('/Users/CPinkston/Documents/Zipfian/Project/Wavs')
    test = build_feature_matrix('/Users/CPinkston/Documents/Zipfian/Project/TestWavs')
    df = pd.DataFrame(test)
    y = df.pop('type')
    print model.predict(df.values)

