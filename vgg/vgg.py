# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from tensorflow import keras
from keras import Input, Sequential
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPool2D

# Provided via: https://raw.githubusercontent.com/mbrukman/reimplementing-ml-papers/main/alexnet/local_response_normalization.py
from local_response_normalization import LocalResponseNormalization


def Conv(filters: int, kernel_size: int, **kwargs) -> Conv2D:
    """Shorthand for defining the Conv2D layers for VGG family of models.

    All VGG models have a stride of 1 and 'same' padding, so to avoid repeating
    these parameters (or skipping them, which leads to 'valid' padding), we use
    a convenience function here.

    Additionally, all hidden layers in VGG are specified to use ReLU activation,
    so we include that here as well.
    """
    return Conv2D(filters, kernel_size, strides=(1, 1), padding='same',
                  activation=keras.activations.relu, **kwargs)


def MaxPool(**kwargs) -> MaxPool2D:
    """Shorthand for defining the MaxPool layers for VGG fmaily of models.

    All pooling layers in VGG are using kernel size of 2 with a stride of 2, so
    we define a shorthand here to ensure their uniformity.
    """
    return MaxPool2D(pool_size=(2, 2), strides=(2, 2), **kwargs)


# Available model types.
MODEL_A = 'A'
MODEL_A_LRN = 'A-LRN'
MODEL_B = 'B'
MODEL_C = 'C'
MODEL_D = 'D'
MODEL_E = 'E'


def VGG(model: str) -> Sequential:
    """Defines a specific VGG model, given one of the valid model types."""
    assert model in (MODEL_A, MODEL_A_LRN, MODEL_B, MODEL_C, MODEL_D, MODEL_E)

    vgg = Sequential([
        Input(shape=(224, 224, 3)),
    ], name=f'VGG-{model}')

    # First block
    vgg.add(Conv(64, 3, name='Conv2D_1_1'))
    if model == MODEL_A:
        # No other layers are added here.
        pass
    elif model == MODEL_A_LRN:
        vgg.add(LocalResponseNormalization(name='LRN'))
    else:
        vgg.add(Conv(64, 3, name='Conv2D_1_2'))

    vgg.add(MaxPool(name='MaxPool_1'))

    # Second block
    vgg.add(Conv(128, 3, name='Conv2D_2_1'))
    if model in (MODEL_B, MODEL_C, MODEL_D, MODEL_E):
        vgg.add(Conv(128, 3, name='Conv2D_2_2'))

    vgg.add(MaxPool(name='MaxPool_2'))

    # Third block
    vgg.add(Conv(256, 3, name='Conv2D_3_1'))
    vgg.add(Conv(256, 3, name='Conv2D_3_2'))

    if model == MODEL_C:
        vgg.add(Conv(256, 1, name='Conv2D_3_3'))
    elif model in (MODEL_D, MODEL_E):
        vgg.add(Conv(256, 3, name='Conv2D_3_3'))

    # Model E gets an extra layer.
    if model == MODEL_E:
        vgg.add(Conv(256, 3, name='Conv2D_3_4'))

    vgg.add(MaxPool(name='MaxPool_3'))

    # Fourth block
    vgg.add(Conv(512, 3, name='Conv2D_4_1'))
    vgg.add(Conv(512, 3, name='Conv2D_4_2'))

    if model == MODEL_C:
        vgg.add(Conv(512, 1, name='Conv2D_4_3'))
    elif model in (MODEL_D, MODEL_E):
        vgg.add(Conv(512, 3, name='Conv2D_4_4'))

    # Model E gets an extra layer.
    if model == MODEL_E:
        vgg.add(Conv(512, 3, name='Conv2D_4_5'))

    vgg.add(MaxPool(name='MaxPool_4'))

    # Fifth block
    vgg.add(Conv(512, 3, name='Conv2D_5_1'))
    vgg.add(Conv(512, 3, name='Conv2D_5_2'))
    if model == MODEL_C:
        vgg.add(Conv(512, 1, name='Conv2D_5_3'))
    elif model in (MODEL_D, MODEL_E):
        vgg.add(Conv(512, 3, name='Conv2D_5_3'))

    # Model E gets an extra layer.
    if model == MODEL_E:
        vgg.add(Conv(512, 3, name='Conv2D_5_4'))

    vgg.add(MaxPool(name='MaxPool_5'))

    vgg.add(Flatten(name='Flatten'))
    vgg.add(Dense(4096, name='FC_1',
                  activation=keras.activations.relu,
                  kernel_regularizer=keras.regularizers.L2(0.0005)))
    vgg.add(Dropout(0.5, name='FC_1_dropout'))
    vgg.add(Dense(4096, name='FC_2',
                  activation=keras.activations.relu,
                  kernel_regularizer=keras.regularizers.L2(0.0005)))
    vgg.add(Dropout(0.5, name='FC_2_dropout'))
    vgg.add(Dense(1000, name='FC_3', activation=keras.activations.relu))
    vgg.add(Activation(keras.activations.softmax, name='Softmax'))

    return vgg


def VGG_A():
    """Constructs VGG model variant A."""
    return VGG(MODEL_A)


def VGG_A_LRN():
    """Constructs VGG model variant A with LocalResponseNormalization."""
    return VGG(MODEL_A)


def VGG_B():
    """Constructs VGG model variant B."""
    return VGG(MODEL_B)


def VGG_C():
    """Constructs VGG model variant C."""
    return VGG(MODEL_C)


def VGG_D():
    """Constructs VGG model variant D."""
    return VGG(MODEL_D)


def VGG_E():
    """Constructs VGG model variant E."""
    return VGG(MODEL_E)
