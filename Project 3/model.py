# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 08:26:52 2017

@author: Greg McCluskey
"""

import csv
import cv2
import numpy as np

windows = False
#windows = True
forwardslash = True 
bFirst = True

lines = []

if windows: #for testing local only 
    with open('c:/udacity/data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if bFirst:
                bFirst = False
            else:    
                lines.append(line)
#for working on Linux                
else:
    #data given by the class in data.zip. Has successful driving of track 1 
    #forward and backwards.
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if bFirst:
                bFirst = False
            else:    
                lines.append(line)                
    #Has curves training data            
    with open('./data/driving_log2.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    #misc extra data        
    with open('./data/driving_log3.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)   
    #extra bridge data after seeing car fail on the bridge       
    with open('./data/driving_log4.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    #extra dirt corner data         
    with open('./data/driving_log5.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    #extra first turn data and bridge data        
    with open('./data/driving_log6.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)  
    #extra right turn data        
    with open('./data/driving_log7.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)          
                
print("len", len(lines))


from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#split data here
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


#generator
def generator(samples, batch_size_in=32):
    num_samples = len(samples)
    #divide by six to get the correct output batch size.
    #3 images input composed of center, right, left and each one flipped for 
    #at total of 6.  Flipped each image to get a balanced set of data.
    batch_size = batch_size_in // 6 
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                #for each angle... 
                for i in range(3):                        
                            
                    #change backslashes to forward slashes since training was done 
                    #from a windows pc 
                    batch_sample[i] = batch_sample[i].replace('\\', '/')
            
                    name = './data/IMG/'+batch_sample[i].split('/')[-1]
                      
                    
                    #read in image from file
                    image = cv2.imread(name)
                    
                    
                    #augment measurements based on position of camera
                    if i == 0: #center
                        measurement = float(batch_sample[3])                        
                    elif i == 1: #left - bump up
                        measurement = float(batch_sample[3]) + 0.25                        
                    else: #right - subtract
                        measurement = float(batch_sample[3]) - 0.25
                                           
                    #flip =  np.random.choice([True, False])                      
                    flip = True
                    if flip:                        
                        #flip it
                        image_flipped = np.fliplr(image)
                        images.append(image_flipped)
                        measurement_flipped = -measurement
                        angles.append(measurement_flipped)
                    else:    
                        images.append(image)
                        angles.append(measurement)
                
            

            # convert to np arrays for model
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
            
            
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size_in=600)
validation_generator = generator(validation_samples, batch_size_in=128)     
        


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Lambda, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D


#Approximate NVDIA model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
#Dropout made my model worse so it is commented out
#model.add(Dropout(0.2, input_shape=(160.320,3)))
#Ignore sky and dashboard data by cropping to 70 and bottom 25 pixels
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

epochs = 10

model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples), 
          validation_data=validation_generator, 
          nb_val_samples=len(validation_samples), nb_epoch=epochs, verbose=1)



model.save('model.h5')
print("Model saved")

#Save off history to run on local system since AWS Linux does not show graphics. 
import _pickle as cPickle
cPickle.dump(history_object.history['loss'], open('hist_loss.p', 'wb')) 
cPickle.dump(history_object.history['val_loss'], open('hist_val_loss.p', 'wb')) 
print(history_object.history.keys())
print("history data saved")


