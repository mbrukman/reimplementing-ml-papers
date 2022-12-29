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


def AlexNet() -> Sequential:
    return Sequential([
        Input(shape=(227, 227, 3)),
        Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', activation='relu', name='Conv1'),
        LocalResponseNormalization(name='LRN1'),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='MaxPool1'),
        Conv2D(filters=256, kernel_size=(5, 5), padding='same', activation='relu', name='Conv2'),
        LocalResponseNormalization(name='LRN2'),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='MaxPool2'),
        Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu', name='Conv3'),
        Conv2D(filters=384, kernel_size=(3, 3), padding='same', activation='relu', name='Conv4'),
        Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', name='Conv5'),
        MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid', name='MaxPool3'),
        Flatten(name='Flatten'),
        Dense(4096, activation='relu', name='Dense1'),
        Dropout(0.5, name='Dropout1'),
        Dense(4096, activation='relu', name='Dense2'),
        Dropout(0.5, name='Dropout2'),
        Dense(1000, activation='softmax', name='Output'),
    ], name='AlexNet')
