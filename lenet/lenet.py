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

import tensorflow as tf
from tensorflow import keras
from keras import Input, Sequential
from keras.layers import Activation, AveragePooling2D, Conv2D, Dense, Flatten, Layer, MaxPooling2D

from typing import Callable, Type


def LeNet(subsampling: Type[keras.layers.Layer] = AveragePooling2D,
          activation: Callable[[tf.Tensor], tf.Tensor] = keras.activations.tanh) -> Sequential:
    return Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation=activation, name='C1'),
        subsampling(pool_size=(2, 2), strides=(2, 2), name='S2'),
        Activation(activation, name='S2_act'),
        Conv2D(filters=16, kernel_size=(5, 5), activation=activation, name='C3'),
        subsampling(pool_size=(2, 2), strides=(2, 2), name='S4'),
        Activation(activation, name='S4_act'),
        Conv2D(filters=120, kernel_size=(5, 5), activation=activation, name='C5'),
        Flatten(name='Flatten'),
        Dense(84, activation=activation, name='F6'),
        Dense(10, activation=keras.activations.softmax, name='Output'),
    ], name='LeNet-5')
