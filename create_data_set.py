import pyaudio
import numpy as np
from scipy import fft

#Atributes for the stream    
NUM_SAMPLES = 1024
SAMPLING_RATE = 22050 #11025
SPECTROGRAM_LENGTH = 50


def get_sample():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLING_RATE,
                     input=True, frames_per_buffer=NUM_SAMPLES)
    audio_data  = np.fromstring(stream.read(NUM_SAMPLES), dtype=np.short)
    stream.close()
    normalized_data = audio_data / 32768.0
    sound_fft = abs(fft(normalized_data))[:NUM_SAMPLES/2]
    
    bucket_scores=[]
    mels = np.array([1,200,400,630,920,1270,1720,2320,3200])/6.25
    
    for i,x in enumerate(mels):
        if i == 0:
            continue
        else:
            bucket_scores.append(sum(sound_fft[mels[i-1]:x]))
    
    if bucket_scores[0] > 10:
        print bucket_scores
        
if __name__ == '__main__':
    while 1==1:
        get_sample()
