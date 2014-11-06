##Visualizing digital signal processing in real time

The goal for this project is to visualize the human voice and give guidance to those seeking to train their voice. The program will take in an argument of the sound that the user is attempting to replicate and output a static bar barplot of the formants that they are attempting to replicate. They can then speak into their microphone to control a bar graph plotted over the top of the target sound to train their voice. The graph should output in real time the sound their voice is most like and give an overview of the sound probabilities when the program completes.

## Current state
Right now the program has been built to output the target graph and return predictions and probabilities of test files located on my computer. To make this work you will need a directory of training and test data as well as their locations entered into the computer. When the model is complete the classifier will be pickled and the program will be able to take in a new voice as a test value.
