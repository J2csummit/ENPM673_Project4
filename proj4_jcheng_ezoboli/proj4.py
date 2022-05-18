#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  1 20:55:16 2022

@author: justincheng
"""

from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from keras.models import model_from_json
import os

# Configure setting with CPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure Model
batch_size = 50
img_width, img_height, img_num_channels = 32, 32, 3
loss_function = sparse_categorical_crossentropy
no_classes = 100
no_epochs = 100
optimizer = Adam()
validation_split = 0.2
verbosity = 1

# Load CIFAR-100 data
(input_train, target_train), (input_test, target_test) = cifar100.load_data()

# Determine Convolution Pool shape
input_shape = (img_width, img_height, img_num_channels)

# Cast as float
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Data Normalization
input_train = input_train / 255
input_test = input_test / 255

# CNN Model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(no_classes, activation='softmax'))

# Compile and fit model
model.compile(loss=loss_function,
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit(input_train, target_train,
            batch_size=batch_size,
            epochs=no_epochs,
            verbose=verbosity,
            validation_split=validation_split)

# Evaluate metrics
score = model.evaluate(input_test, target_test, verbose=0)
print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
 
# Convert model to JSON
model_json = model.to_json()
with open("own_model.json", "w") as json_file:
    json_file.write(model_json)
    
# Convert weights to HDF5
model.save_weights("own_model.h5")
print("Saved model to disk")