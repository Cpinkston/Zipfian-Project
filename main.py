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
import matplotlib.animation as animation

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
    max_value = max(sound_fft)
    print max_value
    sound_fft = sound_fft/max_value
    
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
            if a_file == '.DS_Store':
                break
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

    forest = RandomForestClassifier(n_estimators=100,criterion='entropy', max_features='log2', oob_score=True)
    forest.fit(df.values, y.values)
    return forest

def build_feature_bar(folder, target = 'A'):
    '''
    Input: A String containing the location of your training data,
           A string that corresponds to the target for the graph 

    Output: None, Builds a matplotlib graph

    This builds the background of a matplotlib graph that will show a particular sound
    '''

    sound_data = build_feature_matrix(folder)
    df = pd.DataFrame(sound_data)
    df = df[df['type'] == target]
    y = df.pop('type')

    aggre_A = np.mean(df.values, axis = 0)

    plt.bar([0,1,2,3,4,5,6,7,8,9], aggre_A, width=0.8)
    plt.show()
    plt.title('Aggregate A')



if __name__ == "__main__":
    model = fit_model('/Users/CPinkston/Documents/Zipfian/Project/Wavs')
    test = build_feature_matrix('/Users/CPinkston/Documents/Zipfian/Project/TestWavs')
    df = pd.DataFrame(test)
    y = df.pop('type')
    print model.predict(df.values)
    print model.predict_proba(df.values)

    build_feature_bar('/Users/CPinkston/Documents/Zipfian/Project/Wavs', target='U')
