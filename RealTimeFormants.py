import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pyaudio
import struct
import wave
from scipy.linalg import norm
 
 
SAVE = 0.0
TITLE = ''
FPS = 25.0
 
nFFT = 512
BUF_SIZE = 4 * nFFT
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
 
mels = [1,200,400,630,920,1270,1720,2320,3200,6400]
 
def animate(i, line, stream, MAX_y):
    '''
    Input: The line is the numpy array of data that is being plotted,
           The stream is a pyaudio stream
           Max_y is the max y value

    Output: A new list of data to be plotted on the graph

    This is the animation function that is accepted matplotlib.animation 
    It does the computation and outputs new values for the graph
    '''
    N = max(stream.get_read_available() / nFFT, 1) * nFFT
    data = stream.read(N)

    y = np.array(struct.unpack("%dh" % (N * CHANNELS), data)) / MAX_y
  
    sound_fft = np.fft.fft(y)
    sound_fft = abs(sound_fft[:(len(sound_fft)/2)])
    norm_value = norm(sound_fft)
    sound_fft = sound_fft/norm_value
      
    bucket_scores=[]

    for i,x in enumerate(mels):
        if i == 0:
            continue
        else:
            name = 'bucket'+str(i)
            bucket_scores.append(sum(sound_fft[mels[i-1]:x]))

    line.set_ydata(bucket_scores)
    return bucket_scores,

def init(line):
    '''
    Input: line are the intial values that you will be plotting

    Output: A list of zeroes that is equal to the number of buckets 
            for the formants

    This funtion is used to start off the matplotlib animation function

    '''
 
    bucket_scores=[]

    for i,x in enumerate(mels):
        if i == 0:
            continue
        else:
            name = 'bucket'+str(i)
            bucket_scores.append(0)
    return bucket_scores,

def main():
    '''
    Inputs: None

    Outputs: None

    This puts everything together and runs the animation so that
    it accepts a voice as the stream and outputs the formant values 
    in a plot
    '''

    fig = plt.figure()

    x_f = 1.0 * np.arange(9)
    ax = fig.add_subplot(111, title=TITLE, xlim=(x_f[0], x_f[-1]), ylim=(0, 8))
    ax.set_yscale('symlog', linthreshy=nFFT**0.5)
 
    line, = ax.plot(x_f, np.zeros(9))

    p = pyaudio.PyAudio()

    frames = None

    MAX_y = 2.0**(p.get_sample_size(FORMAT) * 8 - 1)

    stream = p.open(format=FORMAT, channels=CHANNELS,rate=RATE,\
        input=True,frames_per_buffer=BUF_SIZE)

    ani = animation.FuncAnimation(fig, animate, frames,\
        init_func=lambda: init(line), fargs=(line, stream, MAX_y),\
        interval=1000.0/FPS, blit=False)

    plt.show()
    stream.stop_stream()
    stream.close()
    p.terminate()

if __name__ == '__main__':
    main()





















