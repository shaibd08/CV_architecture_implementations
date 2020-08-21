"""
AlexNet won the 2012 ImageNet ILSVRC challenge, achieving a top-five error of 17%.
Developed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton.
The first to stack convolutional layers on top of one another, instead of stacking a pooling layer on
each convolutional layer.
"""
import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential()
# C1
model.add(keras.layers.Conv2D(96,
                              kernel_size=11,
                              strides=4,
                              padding='valid',
                              input_shape=[227, 227, 3],
                              activation='relu'))
# LRN
model.add(keras.layers.Lambda(lambda inp: tf.nn.local_response_normalization(inp,
                                                                             depth_radius=2,
                                                                             alpha=0.00002,
                                                                             beta=0.75,
                                                                             bias=1)))
# S2
model.add(keras.layers.MaxPool2D(pool_size=3,
                                 strides=2,
                                 padding='valid'))
# C3
model.add(keras.layers.Conv2D(256,
                              kernel_size=5,
                              strides=1,
                              padding='same',
                              activation='relu'))
# LRN
model.add(keras.layers.Lambda(lambda inp: tf.nn.local_response_normalization(inp,
                                                                             depth_radius=2,
                                                                             alpha=0.00002,
                                                                             beta=0.75,
                                                                             bias=1)))
# S4
model.add(keras.layers.MaxPool2D(pool_size=3,
                                 strides=2,
                                 padding='valid'))
# C5 - C7
for filters in [384] * 2 + [256] * 1:
    model.add(keras.layers.Conv2D(filters,
                                  kernel_size=3,
                                  strides=1,
                                  padding='same',
                                  activation='relu'))
    model.add(keras.layers.BatchNormalization())
# S8
model.add(keras.layers.MaxPool2D(pool_size=3,
                                 strides=2,
                                 padding='valid'))
# F9
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.5)) # Dropout
# F10
model.add(keras.layers.Dense(4096, activation='relu'))
model.add(keras.layers.Dropout(0.5)) # Dropout
# Out
model.add(keras.layers.Dense(10,
                             activation='softmax'))
