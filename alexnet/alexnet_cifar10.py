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

from local_response_normalization import LocalResponseNormalization

from tensorflow import keras
from keras import Input, Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPool2D

from typing import Type


def AlexNet(lrn: Type = LocalResponseNormalization,
            lrn_name: str = 'TF-NN-LRN') -> Sequential:
    return Sequential([
        Input(shape=(32, 32, 3)),
        Conv2D(filters=64, kernel_size=5, strides=1, padding='same', activation='relu', name='Conv1'),
        MaxPool2D(pool_size=3, strides=2, padding='valid', name='MaxPool1'),
        lrn(name='LRN1'),
        Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu', name='Conv2'),
        lrn(name='LRN2'),
        MaxPool2D(pool_size=3, strides=2, padding='valid', name='MaxPool2'),
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', name='Local3'),
        Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', name='Local4'),
        Flatten(name='Flatten'),
        Dense(10, activation='softmax', name='FC10'),
    ], name=f'CIFAR-10-{lrn_name}')
