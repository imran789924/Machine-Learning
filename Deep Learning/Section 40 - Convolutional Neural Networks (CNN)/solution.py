#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 16:25:44 2020

@author: imran
"""

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense


classifier = Sequential()

#32 feature detectors, (3*3) dimentional
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation = 'relu' ))
classifier.add(MaxPool2D(pool_size=(2,2)))


classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))



classifier.add(Flatten())


#output_dim is number of nodes in hidden layer.
classifier.add(Dense(output_dim = 128, activation = 'relu'))

classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=15,
                    validation_data=test_set,
                    validation_steps=2000)
