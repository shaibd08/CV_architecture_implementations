"""
LeNet-5 is made up of 7 layers, including 3 convolutional layers, 2 pooling layers, and 2 FC layers.
it was created by Yann LeCun in 1998 and has been widely used for handwritten digit recognition (MNIST)
"""
from tensorflow import keras

model = keras.models.Sequential()
# In + C1
model.add(keras.layers.Conv2D(6,
                              kernel_size=5,
                              input_shape=[32, 32, 1],
                              activation='tanh',
                              strides=1))
# S2
model.add(keras.layers.AveragePooling2D(pool_size=2,
                                        strides=2))
# C3
model.add(keras.layers.Conv2D(16,
                              kernel_size=5,
                              strides=1,
                              input_shape=[14, 14, 6],
                              activation='tanh'))
# S4
model.add(keras.layers.AveragePooling2D(pool_size=2,
                                         strides=2))
# C5
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(120, activation='tanh'))
# F6
model.add(keras.layers.Dense(84,
                             activation='tanh'))
# Out
model.add(keras.layers.Dense(10,
                             activation='softmax'))
