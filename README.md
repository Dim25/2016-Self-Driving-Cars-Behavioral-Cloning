==================================================================
# Files:

model.py - Used to create the model.
train.py - Used to generate data batches and train the model. Saves model and weights into current folder. Try/Except used to catch except "KeyboardInterrupt" and "SystemExit" to save weights and model before exit. Useful for debug.
drive.py - The script to drive the car. Run as "python drive.py model.json".
model.json - The model architecture.
model.h5 - Sample of saved model weights.
"/screenshots" folder contains some of the output examples 

==================================================================
# Introduction: 
The Deep Neural Network are used to predict the stearing angle based on input image from front camera of the car.  
Multiple Neural Networks (including, VGG, simple 3 layer CNN, Nvidia 5 layer CNN [https://arxiv.org/pdf/1604.07316v1.pdf], CommaAI CNN) were tested to comeup with final model wich is based on CommaAI CNN model.

==================================================================
# Structure of network:

The nonlinear CommaAI CNN model is used for steering angle prediction, it consists from three convolution layers:

 8x8 convolution
 5x5 convolution
 5x5 convolution

with 'Exponential Linear Units' (ELUs) for better generalization and faster learning, and,
with two dropout layers: 'dropout 0.20' and 'dropout 0.50' to prevent Neural Network from Overfitting.

Adam optimiser is used with mean squared error (MSE) as a loss function. After the set of experiments the learning rate is set to 0.0001 (lr=0.0001).

==================================================================
# Training approach:

Training data provided in batches (with batch_size=256) to reduce the memory usage. Some of the images are mirrored (flipped right to left with steering angle multiplied by minus one).

Training data consists from multiple recovery laps (2 laps of recovery from each side) and 'normal' driving samples. Recovery data introduced to enable car to recover to normal driving after it performed a command not recorded in normal dataset. 

Data is split into training (85%) and validation (15%) datasets to help diagnose the overfitting/underfitting. Test data isn't necessary for this problem (the actual perfomance on the track is used as a key success metric).

'EarlyStopping' callback is used to stop training when a validation loss has stopped improving (not decreasing for 1+ epochs).
'ModelCheckpoint' callback is used to save the model after every epoch (if validation loss has improved).

Examples with screenshots ("/screenshots" folder):
1. "Screen Shot 2016-12-26 at 12.02.02.png" – after car reached left side of the road the model produces steering angle of '1.77' (turn right) to avoid crossing the yellow line. 
2. "Screen Shot 2016-12-26 at 12.03.03.png" – car got pretty close to the right side of the road, so model outputted the '-7.47' steering angle (turn to the left). 
3. "Screen Shot 2016-12-26 at 12.11.11.png" – car reached to close to the right turn, model outputted the '-10.70' steering angle (sharp turn to the left). 

==================================================================
# How to improve the model further:

1. Extra data – additional data on smooth recovery shall help to avoid some of the zigzagging the model still have.
2. Acceleration Speed - at the moment model performs well with small throttles (0.1 – 0.2) only.
3. Experiment with MaxPooling instead of sub sampling. 

