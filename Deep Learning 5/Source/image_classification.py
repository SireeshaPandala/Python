# Simple CNN model for CIFAR-10
import keras
import numpy as np
from keras.datasets import cifar10
from keras.engine.saving import model_from_json
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
if K.backend()=='tensorflow':
    K.set_image_dim_ordering('th')

import tensorflow as tf
import multiprocessing as mp

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

from matplotlib import pyplot
from scipy.misc import toimage
from keras.datasets import cifar10
def show_imgs(X):
    pyplot.figure(1)
    k = 0
    for i in range(0,2):
        for j in range(0,2):
            pyplot.subplot2grid((2,2),(i,j))
            pyplot.imshow(toimage(X[k]))
            k = k+1
    # show the plot
    pyplot.show()


# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# show_imgs(X_test[:4])

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

#z-score
mean = np.mean(X_train,axis=(0,1,2,3))
std = np.std(X_train,axis=(0,1,2,3))
X_train = (X_train-mean)/(std+1e-7)
X_test = (X_test-mean)/(std+1e-7)

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
# Create the model
model = Sequential()
# first layer
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
# second layer
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# Third layer
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
# fourth layer
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
# fifth layer
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
# Dropout layer at 20%.
model.add(Dropout(0.2))
# sixth layer
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # Flatten layer.
model.add(Dropout(0.2))
# Fully connected layer with 1024units and a rectifier activation function.
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
# Fully connected output layer with 10 units and a softmaxactivation function
model.add(Dense(10, activation='softmax'))
# Compile model
epochs = 1
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    )
datagen.fit(X_train)

print(model.summary())




#training
batch_size = 32

# opt_rms = keras.optimizers.rmsprop(lr=0.001,decay=1e-6) optimizer=opt_rms
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),\
                    steps_per_epoch=(X_train.shape[0] // batch_size),epochs=1,\
                    verbose=1,validation_data=(X_test,y_test),
                    callbacks=[TensorBoard(log_dir="logs/final2", histogram_freq=1, write_graph=True, write_images=True)])


# Fit the model
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32,
#           callbacks=[TensorBoard(log_dir="logs/final2", histogram_freq=1, write_graph=True, write_images=True)])
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)

#save to disk
model_json = model.to_json()
with open('model.json', 'w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')

# model.save('./model1' + '.h5')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.h5')

labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

indices = np.argmax(model.predict(X_test[:4]),1)
print([labels[x] for x in indices])


print('Test loss:', scores[0])
print("Accuracy: %.2f%%" % (scores[1]*100))

show_imgs(X_test[:4])

# Load trained CNN model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)
# model.load_weights('model.h5')
#
# labels =  ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
#
# indices = np.argmax(model.predict(X_test[:4]),1)
# print([labels[x] for x in indices])

keras_callbacks = [
    ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', save_best_only=True, verbose=2),
    ModelCheckpoint('/tmp/keras_checkpoints/model.{epoch:02d}.hdf5', monitor='val_loss', save_best_only=True, verbose=0),
    TensorBoard(log_dir='./model_3', histogram_freq=0, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None),
    EarlyStopping(monitor='val_mean_absolute_error', patience=20, verbose=0)
]