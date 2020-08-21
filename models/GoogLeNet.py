"""
Created by Christian Szegedy et al. from Google Research.
Won the ILSVRC 2014 challenge. top-five error below 7%.
Introduced Inception modules.
"""
import tensorflow as tf
from tensorflow import keras


def inception_module(inp, filters1x1, filters1x1_to_3x3, filters3x3, filters1x1_to_5x5, filters5x5,
                     filters1x1_from_pool, activation='relu'):
    """
    Inception module. Acting as a “multi-level feature extractor” by computing
    1×1, 3×3, and 5×5 convolutions within the same module, and stacking the output
    along the channel dimension.
    :param inp: output from previous layer
    :param filters1x1: number of filters for conv_1x1
    :param filters1x1_to_3x3: number of filters for conv_1x1_2
    :param filters3x3: number of filters for conv_3x3
    :param filters1x1_to_5x5: number of filters for conv_1x1_3
    :param filters5x5: number of filters for conv_5x5
    :param filters1x1_from_pool: number of filters for conv_1x1_4
    :param activation: activation function. Default is ReLU.
    :return:
    """
    # conv 1x1+1(S)
    conv_1x1 = keras.layers.Conv2D(filters=filters1x1, kernel_size=1, strides=1, padding='same', activation=activation)(
        inp)
    # conv 1x1+1(S), conv 3x3+1(S)
    conv_1x1_2 = keras.layers.Conv2D(filters=filters1x1_to_3x3, kernel_size=1, strides=1, padding='same',
                                     activation=activation)(inp)
    conv_3x3 = keras.layers.Conv2D(filters=filters3x3, kernel_size=3, strides=1, padding='same', activation=activation)(
        conv_1x1_2)
    # conv 1x1+1(S), conv 5x5+1(S)
    conv_1x1_3 = keras.layers.Conv2D(filters=filters1x1_to_5x5, kernel_size=1, strides=1, padding='same',
                                     activation=activation)(inp)
    conv_5x5 = keras.layers.Conv2D(filters=filters5x5, kernel_size=5, strides=1, padding='same', activation=activation)(
        conv_1x1_3)
    # maxpool 3x3+1(S), conv 1x1+1(S)
    maxpool = keras.layers.MaxPool2D(pool_size=3, strides=1, padding='same')(inp)
    conv_1x1_4 = keras.layers.Conv2D(filters=filters1x1_from_pool, kernel_size=1, strides=1, padding='same',
                                     activation=activation)(maxpool)
    # depth concatenation of the outputs
    output = keras.layers.concatenate([conv_1x1, conv_3x3, conv_5x5, conv_1x1_4], axis=3)
    return output


def googlenet(activation='relu'):
    """
    building the GoogLeNet architecture
    :param activation: activation function. Default is ReLU
    :return: GoogLeNet model
    """
    inp = keras.layers.Input(shape=(224, 224, 3))
    conv_7x7_2s = keras.layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='same', activation=activation)(inp)
    pool_3x3_2s = keras.layers.MaxPool2D(pool_size=3, strides=2)(conv_7x7_2s)
    lrn1 = keras.layers.Lambda(
        lambda x: tf.nn.local_response_normalization(x, depth_radius=2, alpha=0.00002, beta=0.75, bias=1))(pool_3x3_2s)
    conv_1x1_1s = keras.layers.Conv2D(filters=64, kernel_size=1, strides=1, padding='same', activation=activation)(lrn1)
    conv_3x3_1s = keras.layers.Conv2D(filters=192, kernel_size=3, strides=1, padding='same', activation=activation)(
        conv_1x1_1s)
    lrn2 = keras.layers.Lambda(
        lambda x: tf.nn.local_response_normalization(x, depth_radius=2, alpha=0.00002, beta=0.75, bias=1))(conv_3x3_1s)
    pool_3x3_2s = keras.layers.MaxPool2D(pool_size=3, strides=2)(lrn2)
    inception1 = inception_module(pool_3x3_2s, filters1x1=64, filters1x1_to_3x3=96, filters3x3=128,
                                  filters1x1_to_5x5=16, filters5x5=32, filters1x1_from_pool=32, activation=activation)
    inception2 = inception_module(inception1, filters1x1=128, filters1x1_to_3x3=128, filters3x3=192,
                                  filters1x1_to_5x5=32, filters5x5=96, filters1x1_from_pool=64, activation=activation)
    pool_3x3_2s = keras.layers.MaxPool2D(pool_size=3, strides=2)(inception2)
    inception3 = inception_module(pool_3x3_2s, filters1x1=192, filters1x1_to_3x3=96, filters3x3=208,
                                  filters1x1_to_5x5=16, filters5x5=48, filters1x1_from_pool=64, activation=activation)
    inception4 = inception_module(inception3, filters1x1=160, filters1x1_to_3x3=112, filters3x3=224,
                                  filters1x1_to_5x5=24, filters5x5=64, filters1x1_from_pool=64, activation=activation)
    inception5 = inception_module(inception4, filters1x1=128, filters1x1_to_3x3=128, filters3x3=256,
                                  filters1x1_to_5x5=24, filters5x5=64, filters1x1_from_pool=64, activation=activation)
    inception6 = inception_module(inception5, filters1x1=112, filters1x1_to_3x3=144, filters3x3=288,
                                  filters1x1_to_5x5=32, filters5x5=64, filters1x1_from_pool=64, activation=activation)
    inception7 = inception_module(inception6, filters1x1=256, filters1x1_to_3x3=160, filters3x3=320,
                                  filters1x1_to_5x5=32, filters5x5=128, filters1x1_from_pool=128, activation=activation)
    pool_3x3_2s = keras.layers.MaxPool2D(pool_size=3, strides=2)(inception7)
    inception8 = inception_module(pool_3x3_2s, filters1x1=256, filters1x1_to_3x3=160, filters3x3=320,
                                  filters1x1_to_5x5=32, filters5x5=128, filters1x1_from_pool=128, activation=activation)
    inception9 = inception_module(inception8, filters1x1=384, filters1x1_to_3x3=192, filters3x3=384,
                                  filters1x1_to_5x5=48, filters5x5=128, filters1x1_from_pool=128, activation=activation)
    global_avg_pool = keras.layers.GlobalAvgPool2D()(inception9)
    flat = keras.layers.Flatten()(global_avg_pool)
    dropout = keras.layers.Dropout(0.4)(flat)
    classification_layer = keras.layers.Dense(1000, activation='softmax')(dropout)

    model = keras.models.Model(inputs=inp, outputs=classification_layer)

    return model

# GoogLeNet model
model = googlenet()
# model.summary()