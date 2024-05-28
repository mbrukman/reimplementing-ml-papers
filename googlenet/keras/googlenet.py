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

from __future__ import annotations
from typing import Callable, Optional, List, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from keras import Input, Model, Sequential
from keras.layers import Activation, AvgPool2D, Concatenate, Conv2D, Dense, Dropout, Flatten, Layer, MaxPool2D

from local_response_normalization import LocalResponseNormalization

class Inception(Layer):
    filters_1x1: int
    filters_1x1_reduce_3x3: int
    filters_3x3: int
    filters_1x1_reduce_5x5: int
    filters_5x5: int
    pool_proj: int
    module_name: str

    conv_1x1: Conv2D
    conv_1x1_3x3: Sequential
    conv_1x1_5x5: Sequential
    max_pool_conv: Sequential

    def __init__(self,
                 filters_1x1: int,
                 filters_1x1_reduce_3x3: int,
                 filters_3x3: int,
                 filters_1x1_reduce_5x5: int,
                 filters_5x5: int,
                 pool_proj: int,
                 name: str,
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.filters_1x1 = filters_1x1
        self.filters_1x1_reduce_3x3 = filters_1x1_reduce_3x3
        self.filters_3x3 = filters_3x3
        self.filters_1x1_reduce_5x5 = filters_1x1_reduce_5x5
        self.filters_5x5 = filters_5x5
        self.pool_proj = pool_proj
        self.module_name = name

    def _conv2d(self, filters: int, kernel_size: int, name: str) -> Conv2D:
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      padding='same', activation='relu',
                      name=f'{self.module_name}_{name}')

    def build(
        self, input_shape: Union[List[Optional[int]],
                                 Tuple[Optional[int], int, int, int]]) -> None:
        """Builds internal structures to prepare for model training."""
        self.conv_1x1 = self._conv2d(self.filters_1x1, 1, 'Conv_1x1')

        self.conv_1x1_3x3 = Sequential([
            self._conv2d(self.filters_1x1_reduce_3x3, 1, 'Conv_1x1_3x3'),
            self._conv2d(self.filters_3x3, 3, 'Conv_3x3'),
        ])

        self.conv_1x1_5x5 = Sequential([
            self._conv2d(self.filters_1x1_reduce_5x5, 1, 'Conv_1x1_5x5'),
            self._conv2d(self.filters_5x5, 5, 'Conv_5x5'),
        ])

        self.max_pool_conv = Sequential([
            MaxPool2D(3, 1, padding='same', name=f"{self.module_name}_MaxPool"),
            self._conv2d(self.pool_proj, 1, 'MaxPool_Conv_1x1'),
        ])

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return Concatenate(axis=-1)([
            self.conv_1x1(inputs),
            self.conv_1x1_3x3(inputs),
            self.conv_1x1_5x5(inputs),
            self.max_pool_conv(inputs),
        ])


def SequentialPassthrough(layers: List[Layer]) -> Callable[[tf.Tensor], tf.Tensor]:
    """Similar to Keras' `Sequential`, but shows all layers transparently.

    Instead of hiding all the layers behind another abstraction called
    `Sequential`, this function explicitly shows all the layers involved in the
    model, so they're visible when calling `model.summary()`.
    """
    def process_layers(input_: tf.Tensor) -> tf.Tensor:
        x = input_
        for layer in layers:
          x = layer(x)
        return x

    return process_layers

def GoogLeNet() -> Model:
    """GoogLeNet model implementation."""

    input_: Input = Input(shape=(224, 224, 3), name='Input')

    x = SequentialPassthrough([
        Conv2D(64, 7, 2, activation='relu', padding='same', name='Conv1'),
        MaxPool2D(3, 2, padding='same', name='MaxPool_1'),
        LocalResponseNormalization(name='LRN1'),
        Conv2D(192, 1, activation='relu', padding='valid', name='Conv_2'),
        Conv2D(192, 3, activation='relu', padding='same', name='Conv_3'),
        LocalResponseNormalization(name='LRN2'),
        MaxPool2D(3, 2, padding='same', name='MaxPool_2'),
        Inception(64, 96, 128, 16, 32, 32, name='Inception_3a'),
        Inception(128, 128, 192, 32, 96, 64, name='Inception_3b'),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same', name='MaxPool_3'),
        Inception(192, 96, 208, 16, 48, 64, name='Inception_4a'),
    ])(input_)

    # Output 0 branch
    output0 = SequentialPassthrough([
        AvgPool2D(5, 3, padding='valid', name='AvgPool_out0'),
        Conv2D(128, 1, padding='same', activation='relu', name='Conv2D_out0'),
        Flatten(name='Flatten_out0'),
        Dense(1000, activation='relu', name='FC_1_out0'), ## params
        Dropout(0.7, name='Dropout_out0'),
        Dense(1000, activation='relu', name='FC_2_out0'), ## params
        Activation('softmax', name='Activation_out0'),
    ])(x)

    # Continue with more Inception modules
    y = SequentialPassthrough([
        Inception(160, 112, 224, 24, 64, 64, name='Inception_4b'),
        Inception(128, 128, 256, 24, 64, 64, name='Inception_4c'),
        Inception(112, 144, 288, 32, 96, 64, name='Inception_4d'),
    ])(x)

    # Output 1 branch
    output1 = SequentialPassthrough([
        AvgPool2D(5, 3, padding='valid', name='AvgPool_out1'),
        Conv2D(128, 1, padding='same', activation='relu', name='Conv2D_out1'),
        Flatten(name='Flatten_out1'),
        Dense(1000, activation='relu', name='FC_1_out1'), ## params
        Dropout(0.7, name='Dropout_out1'),
        Dense(1000, activation='relu', name='FC_2_out1'), ## params
        Activation('softmax', name='Activation_out1'),
    ])(y)

    # Continue with more Inception modules
    output2 = SequentialPassthrough([
        Inception(256, 160, 320, 32, 128, 128, name='Inception_4e'),
        MaxPool2D(3, 2, padding='same', name='MaxPool_4'),
        Inception(256, 160, 320, 32, 128, 128, name='Inception_5a'),
        Inception(384, 192, 384, 48, 128, 128, name='Inception_5b'),
        AvgPool2D(7, padding='valid', name='AvgPool_out2'),
        Flatten(name='Flatten_out2'),
        Dropout(0.4, name='Dropout_out2'),
        Dense(1000, activation='relu', name='FC_out2'),
        Activation('softmax', name='Activation_out2'),
    ])(y)

    return Model(inputs=input_, outputs=[output0, output1, output2], name='GoogLeNet')
