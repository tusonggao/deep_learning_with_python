from __future__ import print_function, division, with_statement
import os
import sys
import time
import numpy as np
import pandas as pd

import keras

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('train_images.shape is ', train_images.shape, 'len of train_labels is ', len(train_labels))
print('test_images.shape is ', test_images.shape, 'len of test_labels is ', len(test_labels))


