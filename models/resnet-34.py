"""
ResNet-34 has 34 layers (only counting convolutional layers and the fully connected layers containing 3 residual units
that output 64 feature maps, 4 ResidualUnitLayers with 128 maps, 6 ResidualUnitLayers with 256 maps, and 3
ResidualUnitLayers with 512 maps.
"""
from tensorflow import keras


class ResidualUnitLayer(keras.layers.Layer):
    """
    Residual block subclass.
    """
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters=filters,
                                kernel_size=3,
                                strides=strides,
                                padding='SAME',
                                use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters=filters,
                                kernel_size=3,
                                strides=1,
                                padding='SAME',
                                use_bias=False),
            keras.layers.BatchNormalization()]
        self.skip_layers = []
        # skip connection when changing feature map size and depth
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters=filters,
                                    kernel_size=1,
                                    strides=strides,
                                    padding='SAME',
                                    use_bias=False),
                keras.layers.BatchNormalization()]

    def call(self, inputs, **kwargs):
        """
        the input goes through the main layers and skip layers, which are then added and passed through the activation
        function
        """
        z = inputs
        for layer in self.main_layers:
            z = layer(z)
        skip_z = inputs
        for layer in self.skip_layers:
            skip_z = layer(skip_z)
        return self.activation(z + skip_z)


model = keras.models.Sequential()
# C1
model.add(keras.layers.Conv2D(64,
                              kernel_size=7,
                              input_shape=[224,224,3],
                              strides=2,
                              padding='SAME',
                              use_bias=False))
# BN
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
# Max Pool
model.add(keras.layers.MaxPool2D(pool_size=3,
                                 strides=2,
                                 padding='SAME'))
prev_filters = 64
# Sequence of residual blocks
for filters in [64]*3+[128]*4+[256]*6+[512]*3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnitLayer(filters=filters,
                                strides=strides))
    prev_filters = filters
# Global average pool layer
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
# Classification layer
model.add(keras.layers.Dense(1000,
                             activation='softmax'))

