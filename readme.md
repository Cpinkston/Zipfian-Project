##Visualizing digital signal processing in real time
####Overview
The goal for this project is to visualize the human voice and give guidance to those seeking to train their voice. The current program is trained on 39 arpabet (http://en.wikipedia.org/wiki/Arpabet) phonemes that make up North American English. The program also contains over 130,000 words that can be built out into their spectral envelope for reference.

####The two stages of training a voice
1. Select a sound to create in the plot type drop down box. This will update the top graph giving a reference point of how your voice should look when it is displayed. When you attempt to speak your sound will update over the top of the refernce for you to comare your sound in real time. The program will also give you it's closest guess to the sound that you are creating. When you feel comfortable with a few sounds you can move on to step two

2. Now you can attempt to combine the sounds to create a whole word. When you type a valid word in the text box the target spectral envelope graph will update on the bottom right to show the approximate word silhouette. The sounds for that word will update at the top of the screen. Then you can step through each sound which udate the graph on the bottom right. After you have said the sounds that make up the target word you can compare your sounds to the sillhouette. If there is a sound that seems off take that sound back to step one.

##Files
###VoicePlot.py
This is the current version of the desktop application that I have created to visualize sound. This file requires the enthought's traits and chaco to be run. The enthought python distribution can be downloaded here https://store.enthought.com/. Data files fit to my voice can be found within the data file provided in the data folder. I am currently working on building an executable that will be provided here to download the application.

### data/
This folder contains all the pickled models and example csv files for creating your own datasets on your voice

### create_data_set.py
This file was used to build the backgrounds of the in time sectrogram graph up top. It will output corresponding digits for a sound. You can use these numbers as a dataset to model the particular sound you are making at the time

### dictionary_builder.py
This file takes a dictionary input of a word followed by the syllables that are spoken in their ARPAbet form. All contents should be separated by spaces and new words on separate lines

###main.py
This file was used as my proof of concept that you can predict phonemes using a fast fourier transform and a melodic scale. To set up this file you will need a location on your computer with that has a training WAV folder that contains folders of specific sounds to try and predict on. You will also need a Test WAV folder that contains a second folder that holds your test wav files to predict on. The file locations should be updated within the corresponding locations in the main block. The file will then output it's predicted values and probabilities. The file will also build a formant graph for the specified sound in the build_feature_bar() function.

###model_builder.py
This file will create the predictive models that are used in voiceplot. The data the file corresponds with is the data created in create_data_set.py with the sound you are ARPAbet sound you are creating added to the end. The program takes a location of the csv you have created and a destination location for the pickled file that it outputs.