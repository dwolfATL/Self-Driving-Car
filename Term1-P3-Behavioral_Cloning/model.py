# -*- coding: utf-8 -*-
"""
# Behavioral Cloning with Keras
# Self Driving Cars

@author: DWolf
"""

import pickle
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Lambda, Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

# load dataset
# The data has already been pre-processed including
# including resizing and cropping to 66x200 pixels

pkldata = 'pkldata.p'

with open(pkldata, mode='rb') as f:
    data = pickle.load(f)

X, y = data['features'], data['labels']
    
print("Number of labels:",len(y))
print("Feature array shape:",X.shape)

# number of images
n_img = len(y)
# index where the left camera images begin
l_idx = int(n_img / 3)
# index where the right camera images begin
r_idx = int(l_idx * 2)
print("Index for start of left camera images: ", l_idx)
print("Index for start of right camera images: ", r_idx)
# input image dimensions
img_rows, img_cols, colors = X[0].shape


# preprocess the data 1/3
# (Note that images were already resized and cropped to 50x100 pixels within data_prep.py)

# Adjust the steering angle for left and right camera images (to pretend as if it is a center image)
adj = 0.08
y[l_idx : r_idx-1] =  [x+adj for x in y[l_idx : r_idx-1]] 
y[r_idx : len(y)-1] = [x-adj for x in y[r_idx : len(y)-1]] 

  
# preprocess the data 2/3
# To help with the right turn, implemented a manual adjustment to increase the magnitude of steering angle

# index at start of the right turn center image
r_turn_idx = 5432
# number of frames for the right turn center image
r_turn_len = 40

def manual_angle(y, idx, length, offset):
    for k in range(idx, idx + length):
        y[k] += offset
    for k in range(l_idx + idx, l_idx + idx + length):
        y[k] += offset
    for k in range(r_idx + idx, r_idx + idx + length):
        y[k] += offset
    return y

y = manual_angle(y, r_turn_idx, r_turn_len, 0.25)


# preprocess the data 2.5
# In addition to amplifying the angle, also add several more examples of the right turn to the training data
# I tried to replace images with consecutive zeroes since those are not as important to the training data
# There are 10 occasions with more than 40 consecutive 0 steering angles, shown here by starting pixel (length)
# 0 (51), 933 (72), 1765 (55), 2564 (52), 2620 (50), 3091 (45), 4463 (56), 6582 (48), 6930 (44), 7914 (122)

replace = [r_turn_idx,943,1775,7924,6940,4473,2574,2630,3101,6592,7990,1000,2000,3000,4000,5000]
   
for i in replace:
    for j in range(r_turn_len):
        # center image
        X[i+j] = X[r_turn_idx+j]
        y[i+j] = y[r_turn_idx+j]
        # left image
        X[i+j+l_idx] = X[r_turn_idx+j+l_idx]
        y[i+j+l_idx] = y[r_turn_idx+j+l_idx]
        # right image
        X[i+j+r_idx] = X[r_turn_idx+j+r_idx]
        y[i+j+r_idx] = y[r_turn_idx+j+r_idx]

# preprocess the data 3/3
# flip every other image horizontally (and also change the sign of the steering angle)
j = -1
for i in range(len(y)):
    if j == 1:
        j *= -1
        X[i] = np.fliplr(X[i])
        y[i] = y[i] * -1.
    else:
        j *= -1


# split into training and testing sets
X_train, X_temp, y_train, y_temp = \
    train_test_split(X, y, test_size=0.4, random_state=42)

X_val, X_test, y_val, y_test = \
    train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print('Train feature size:',X_train.shape)
print('Train label size:',y_train.shape)
print('Validation feature size:',X_val.shape)
print('Validation label size:',y_val.shape)
print('Test feature size:',X_test.shape)
print('Test label size:',y_test.shape)


# create data generators for each feature dataset
c = 0.05
train_generator = ImageDataGenerator(
        width_shift_range=c,
        height_shift_range=c,
        shear_range=c,
        zoom_range=c,
        fill_mode='nearest'
)
    
val_generator = ImageDataGenerator(
        width_shift_range=c,
        height_shift_range=c,
        shear_range=c,
        zoom_range=c,
        fill_mode='nearest'
)

test_generator = ImageDataGenerator(
        width_shift_range=c,
        height_shift_range=c,
        shear_range=c,
        zoom_range=c,
        fill_mode='nearest'
)

# fit the imagedatagenerators
train_generator.fit(X_train)
val_generator.fit(X_val)
test_generator.fit(X_test)

# define model parameters
input_shape = (img_rows, img_cols, colors)
pool_size = (2, 2) # size of pooling area for max pooling
kernel_size_conv1 = (8, 8) # convolution kernel size
kernel_size_conv2 = (5, 5)
kernel_size_conv3 = (5, 5)
stride_conv1 = (4, 4) # convolution stride
stride_conv2 = (2, 2)
stride_conv3 = (2, 2)

# define model architecture
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape))

# conv layer 1
model.add(Convolution2D(16, kernel_size_conv1[0], kernel_size_conv1[1], 
                        subsample=stride_conv1, border_mode="same"))
model.add(Activation('relu'))

# conv layer 2
model.add(Convolution2D(32, kernel_size_conv2[0], kernel_size_conv2[1], 
                        subsample=stride_conv2, border_mode="same"))
model.add(Activation('relu'))

# conv layer 3, with max pooling
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Convolution2D(64, kernel_size_conv3[0], kernel_size_conv3[1], 
                        subsample=stride_conv3, border_mode="same"))

model.add(Flatten())

model.add(Dropout(.2))
model.add(Activation('relu'))

# fully connected layer
model.add(Dense(512))
model.add(Dropout(.5))
model.add(Activation('relu'))

# output layer
model.add(Dense(1))
# end of model architecture

# compile the model
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))

# print model architecture
model.summary()

# more parameters for fitting and evaluating
batch_size = 256
nb_epoch = 16
samples_per_epoch = len(X_train)
nb_val_samples = len(X_val)
val_samples = len(X_test)


# fit the network on training data and also apply validation data
model.fit_generator(train_generator.flow(X_train, y_train, batch_size=batch_size),
                    samples_per_epoch=samples_per_epoch, 
                    nb_epoch=nb_epoch, 
                    verbose=1, 
                    validation_data=val_generator.flow(X_val, y_val, batch_size=batch_size), 
                    nb_val_samples=nb_val_samples)


# evaluate the network on the test data
evaluation = model.evaluate_generator(generator=val_generator.flow(X_test, y_test, batch_size=batch_size),
                                      val_samples=val_samples)
print('Test data mse:', evaluation)

# export the model and weights for drive.py
import json
json_string = model.to_json()
with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)

model.save_weights('model.h5')



