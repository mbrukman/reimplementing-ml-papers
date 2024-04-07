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

from data_sequence import DataSequenceWithShuffling, Strategy

import numpy as np


class NumpyStrategy(Strategy):
    def __init__(self):
        super().__init__(range_fn=np.arange, shuffle_fn=np.random.shuffle)


class NumpyDataSequence(DataSequenceWithShuffling):
    """Numpy-based implementation of `keras.utils.Sequence`."""

    def __init__(self, num_items: int, batch_size: int, shuffle: bool = True):
        super().__init__(num_items=num_items, batch_size=batch_size,
                         shuffle=shuffle, strategy=NumpyStrategy())
