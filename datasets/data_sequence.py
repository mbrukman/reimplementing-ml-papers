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

import math
import random
from typing import List, Tuple


class DataSequence:
    def __init__(self, num_items: int, batch_size: int):
        self.num_items = num_items
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(self.num_items / self.batch_size)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        low = self.batch_size * index
        # Cap at the length of the array; the last batch may be a smaller size
        # if the total number of items is not a multiple of `batch_size`.
        high = min(low + self.batch_size, self.num_items)
        return (low, high)


class DataSequenceWithShuffling:
    def __init__(self, num_items: int, batch_size: int, shuffle: bool = False):
        self.num_items = num_items
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = list(range(0, self.num_items))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(self.num_items / self.batch_size)

    def __getitem__(self, index: int) -> Tuple[Tuple[int, int], List[int]]:
        low = self.batch_size * index
        # Cap at the length of the array; the last batch may be a smaller size
        # if the total number of items is not a multiple of `batch_size`.
        high = min(low + self.batch_size, self.num_items)
        positions = self.indexes[low:high]
        return (low, high), positions

    def on_epoch_end(self):
        if self.shuffle:
            # Note: in practice, prefer to use `numpy.random.shuffle` instead.
            random.shuffle(self.indexes)

