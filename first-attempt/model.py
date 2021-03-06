import csv
import cv2
import numpy as np
import pandas as pd

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.core import Activation, Dense, Dropout, Flatten, Lambda

# http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
# Model definition from NVIDIA with modified first layer
def buildModel():
    model = Sequential()
    model.add(Cropping2D(cropping=((60, 0), (20, 80)), input_shape=(160, 320, 3)))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    # Connected layers
    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))
    return model

# display model layout
model = buildModel()
model.summary()

# from primitive setup -- read data from CSV file
lines = []
with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = '../data/IMG/' + filename
    image = cv2.imread(current_path)
    measurement = float(line[3])
    images.append(image)
    measurements.append(measurement)

# training data as numpay array
X_train = np.array(images)
y_train = np.array(measurements)

# Fit model and save to file model.h5
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=10)
model.save('model.h5')
