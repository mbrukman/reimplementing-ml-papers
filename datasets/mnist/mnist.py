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

import numpy as np
from tensorflow import keras

from typing import Tuple

class MNIST:

    x_train_raw_data: np.ndarray
    x_test_raw_data: np.ndarray
    y_train_raw_data: np.ndarray
    y_test_raw_data: np.ndarray
    num_classes: int

    def __init__(self):
        train_data, test_data = keras.datasets.mnist.load_data()
        self.x_train_raw_data, self.y_train_raw_data = train_data
        self.x_test_raw_data, self.y_test_raw_data = test_data
        self.num_classes = 10

    def _scale_custom(self, array: np.ndarray,
                      target_range: Tuple[float, float]) -> np.ndarray:
        lower_bound, upper_bound = target_range
        assert lower_bound < upper_bound, f'range {target_range} must be (low, high) with low < high'
        return array * (upper_bound - lower_bound) + lower_bound

    def x_train_raw(self) -> np.ndarray:
        return self.x_train_raw_data

    def x_train_scale_0_1(self) -> np.ndarray:
        return self.x_train_raw().astype('float32') / 255.0

    def x_train_scale_custom(self, target_range: Tuple[float, float]) -> np.ndarray:
        return self._scale_custom(self.x_train_scale_0_1(), target_range)

    def x_test_raw(self) -> np.ndarray:
        return self.x_test_raw_data

    def x_test_scale_0_1(self) -> np.ndarray:
        return self.x_test_raw_data.astype('float32') / 255.0

    def x_test_scale_custom(self, target_range: Tuple[float, float]) -> np.ndarray:
        return self._scale_custom(self.x_test_scale_0_1(), target_range)

    def y_train_raw(self) -> np.ndarray:
        return self.y_train_raw_data

    def y_train_categorical(self) -> np.ndarray:
        return keras.utils.to_categorical(self.y_train_raw(), self.num_classes)

    def y_test_raw(self) -> np.ndarray:
        return self.y_test_raw_data

    def y_test_categorical(self) -> np.ndarray:
        return keras.utils.to_categorical(self.y_test_raw(), self.num_classes)
