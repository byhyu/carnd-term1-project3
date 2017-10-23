import csv
import cv2
import numpy as np
import sklearn.utils
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Dropout, Activation, Cropping2D
from os import path

IMAGE_FOLDER_PATH = path.join('data','IMG')
LOG_FILE = path.join('data','driving_log.csv')


def resize_fcn(input):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(input, (64, 64))


#### Load Data
samples=[]
with open(LOG_FILE) as csv_file:
    reader=csv.reader(csv_file)
    for line in reader:
        samples.append(line)
del(samples[0])

train_samples, validation_samples = train_test_split(samples, test_size=0.2)


def generator(samples, batch_size=64):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            ANGLE_CORRECTION = 0.2
            for batch_sample in batch_samples:
                for i in range(3):
                    name = path.join(IMAGE_FOLDER_PATH,batch_sample[i].split('/')[-1])
                    image = cv2.imread(name)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image_lr = np.fliplr(image)
                    images.extend((image, image_lr))
                center_angle = float(batch_sample[3])
                left_angle = float(batch_sample[3])+ ANGLE_CORRECTION
                if left_angle>1:
                    left_angle=1
                right_angle = float(batch_sample[3])-ANGLE_CORRECTION
                if right_angle<-1:
                    right_angle=-1
                angles.extend((center_angle,-center_angle,left_angle,-left_angle,right_angle,-right_angle))

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle (X_train, y_train)


train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

# implement, compile and train the model using the generator function

model=Sequential()
model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(160,320,3),output_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,20), (0,0))))
model.add(Lambda(resize_fcn))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Convolution2D(64,3,3,activation='elu'))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dropout(0.6))
model.add(Dense(50))
model.add(Dropout(0.6))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
history_object=model.fit_generator(train_generator,
                    samples_per_epoch=6*len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=7,verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('loss.png')

model.save('model.h5')