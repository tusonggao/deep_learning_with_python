from __future__ import print_function, division, with_statement
import os
import sys
import time
import numpy as np
import pandas as pd

import keras
from keras.datasets import mnist

from keras import models
from keras import layers

from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('train_images.shape is ', train_images.shape, 'train_labels.shape is ', train_labels.shape)
print('test_images.shape is ', test_images.shape, 'test_labels.shape is ', test_labels.shape)
print('type of train_images is ', type(train_images), 'type of train_labels is ', type(train_labels))


network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print('prog starts here!')

train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print('type of train_labels is ', type(train_labels), 'type of test_labels is ', type(test_labels))
print('shape of train_labels is ', train_labels.shape, 'shape of test_labels is ', test_labels.shape)

sys.exit(0)

network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_acc:', test_acc)

print('prog ends here!')




