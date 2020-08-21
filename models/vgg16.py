""""
Runner up in the ILSVRC 2014 challenge. Developed by Karen Simonyan and Andrew Zisserman from the
Visual Geometry Group (VGG) research lab at Oxford University.
Very simple and classical architecture, with 2 or 3 convolutional layers and a pooling layer, then again 2 or 3
convolutional layers and a pooling layer, and so on, plus a final dense network with 2 hidden layers and
the output layer
"""
from tensorflow import keras


def vgg_block(n_convs, filters):
    """
    vgg block
    :param n_convs: number of convolutional layers in block
    :param filters: number of filters
    :return: block of n_convs convolutional layers and a max pool layer
    """
    block = keras.models.Sequential()
    for _ in range(n_convs):
        block.add(keras.layers.Conv2D(filters,
                                      kernel_size=3,
                                      padding='same',
                                      activation='relu'))
    block.add(keras.layers.MaxPool2D(pool_size=2,
                                     strides=2))
    return block


def vgg(conv_architecture, input_shape):
    """
    creates vgg architecture
    :param conv_architecture: list of tuples, where each tuple is (n_convs, filters)
    :param input_shape: image input dimensions
    :return: vgg model
    """
    model = keras.models.Sequential()
    model.add(keras.layers.Input(shape=input_shape))
    for n_convs, filters in conv_architecture:
        model.add(vgg_block(n_convs, filters))
    # FC layers
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model


# vgg16 architecture
vgg16 = ((2, 64), (2, 128), (3, 256), (3, 512), (3, 512))
model = vgg(vgg16, input_shape=(224, 224, 3))
# model.summary()