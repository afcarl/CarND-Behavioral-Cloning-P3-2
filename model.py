import csv
import cv2
import numpy as np
import keras

lines = [] #create arrays
with open('./data/driving_log.csv') as csvfile: #open file
  reader = csv.reader(csvfile) #read csv file
  # Skip header line
  header = next(reader)
  for line in reader:
    lines.append(line)#append line data to list of lines

images = []
measurements = []
for line in lines:
  for i in range(3): #loads up center, left, right images loaded into images array
    source_path = line[i]
    filename = source_path.split('/')[-1]
    current_path = "./data/IMG/" + filename #file path to images
    image = cv2.imread(current_path) #read images
    images.append(image)
  correction = 0.2
  measurement = float(line[3])#cast measurement as float
  measurements.append(measurement)#steering measurement for center image
  measurements.append(measurement+correction)#steering measuremnet for left
  measurements.append(measurement-correction)#steering measurement for right

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    flipped_image = cv2.flip(image, 1) # flip the image
    flipped_measurement = float(measurement) * -1.0
    augmented_images.append(flipped_image)
    augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)#turn to numpy arrays
y_train = np.array(augmented_measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D

model = Sequential() #define a model
#Nvidia self-driving car model architecture
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))# normalize into a range of -0.5 and 0.5
model.add(Cropping2D(cropping=((70,25),(0,0)))) #crop images to isolate road lines
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))# convo layer
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))# convo layer
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))# convo layer
model.add(Convolution2D(64,3,3,activation='relu'))# convo layer
model.add(Convolution2D(64,3,3,activation='relu'))# convo layer
model.add(Flatten()) #add a flatten layer
model.add(Dense(100))
model.add(Dropout(0.5)) #add a Dropout
model.add(Dense(50))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))#single node to predict steering angle

model.compile(optimizer='adam', loss='mse')
model.fit(X_train,y_train,validation_split=0.2,shuffle=True, nb_epoch=10)
model.save('model.2.h5')
