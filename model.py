import os
import csv
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


def read_samples(base_path):
    """
    Read samples from data directory
    """
    samples = []
    path = os.path.join(base_path, 'driving_log.csv')
    with open(path) as csvfile:
      reader = csv.reader(csvfile)
      # Skip header line
      header = next(reader)
      for line in reader:
        samples.append(line)
    return samples


def get_data(base_path, samples, correction=0.25):
    """
    Get data from samples
    """
    images = []
    measurements = []
    for sample in samples:
        # Read steering meaturement

        # Read center
        path_center = os.path.join(base_path, 'IMG', sample[0].split('/')[-1])
        image_center= cv2.imread(path_center)
        steering_center = float(sample[3])

        # Read Left
        path_left = os.path.join(base_path, 'IMG', sample[1].split('/')[-1])
        image_left= cv2.imread(path_left)
        steering_left = steering_center + correction

        # Read right
        path_right = os.path.join(base_path, 'IMG', sample[2].split('/')[-1])
        image_right= cv2.imread(path_right)
        steering_right = steering_center - correction

        # Append data
        images.extend([image_center, image_left, image_right])
        measurements.extend([steering_center, steering_left, steering_right])
    return images, measurements


def augment_data(images, measurements):
    """
    Augment data
    """
    augmented_images = []
    augmented_measurements = []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        flipped_image = cv2.flip(image, 1)
        flipped_measurement = float(measurement) * -1.0
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)
    return augmented_images, augmented_measurements


def get_model(keep_ratio=0.25):
    """
    Get a keras model for self driving
    """
    model = Sequential()
    # Standardize images
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # Crop images
    model.add(Cropping2D(cropping=((70,25),(0,0)))) #crop images to isolate road lines

    # Convolution layer
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    # Convolution layer
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    # Convolution layer
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    # Convolution layer
    model.add(Convolution2D(64,3,3,activation='relu'))
    # Convolution layer
    model.add(Convolution2D(64,3,3,activation='relu'))

    # Flatten layers
    model.add(Flatten())
    model.add(Dense(100))
    # Fully-connected layer with Dropout to avoid over-fitting
    model.add(Dropout(keep_ratio))
    model.add(Dense(50))
    # Fully-connected layer with Dropout to avoid over-fitting
    model.add(Dropout(keep_ratio))
    model.add(Dense(20))
    # Fully-connected layer with Dropout to avoid over-fitting
    model.add(Dropout(keep_ratio))
    model.add(Dense(10))
    model.add(Dense(1))
    return model



##
## main
##
base_path = '../data'

# Read samples
samples = read_samples(base_path)
# Load data
X, y = get_data(base_path, samples, correction=0.25)
# Augment data
X, y = augment_data(X, y)

# Convert to numpy array
X, y = np.array(X), np.array(y)

# Split data to train ones and validation ones
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

# Get and compile a keras model
model = get_model(keep_ratio=0.25)
model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=128,
    nb_epoch=10,
)


# Plot loss of train and validation
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.legend(['loss', 'val_loss'])
plt.xlabel('# Epochs')
plt.ylabel('Loss')
plt.ylim(0.0, np.max(np.array(loss.extend(val_loss))))
plt.show()


# Save model
save_path = 'model.h5'
model.save(save_path)
