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
"""Implements data sequence for implementing fast and correct data loaders."""

# From Python 3.9 and onward, `tuple`, `list` and other collection classes can
# also function as generic class types (see PEP 585).
#
# Once we no longer need to support Python 3.7 or 3.8, we can remove this syntax
# (added in PEP 563) for Python 3.7 and higher.
from __future__ import annotations

import math
import random
from typing import Any, Callable, List, Optional, Sequence, Tuple


# No type annotations for these type aliases, as `TypeAlias` only became
# available starting in Python 3.10 with https://peps.python.org/pep-0613/
#
# We only use a subset of range() functionality because we always call it with 2
# int params, so we don't need to express a more complex type alias.
_Range = Callable[[int, int], Sequence[int]]

_Shuffle = Callable[[List[Any]], None]


class Strategy:
    """Abstracts the process of generating random permutations and shuffling data."""

    range: _Range
    shuffle: _Shuffle

    def __init__(self, range_fn: _Range = range, shuffle_fn: _Shuffle = random.shuffle):
        self.range = range_fn
        self.shuffle = shuffle_fn


class DataSequence:
    """Very simple implementation of `keras.utils.Sequence` without shuffling."""

    num_items: int
    batch_size: int

    def __init__(self, num_items: int, batch_size: int):
        self.num_items = num_items
        self.batch_size = batch_size

    def __len__(self) -> int:
        return math.ceil(self.num_items / self.batch_size)

    def __getitem__(self, index: int) -> Tuple[int, int]:
        low = self.batch_size * index
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, self.num_items)
        return (low, high)


class DataSequenceWithShuffling:
    """Simple implementation of `keras.utils.Sequence` with optional shuffling."""

    num_items: int
    batch_size: int
    should_shuffle: bool
    strategy: Strategy
    indexes: list[int]

    def __init__(self, num_items: int, batch_size: int, shuffle: bool = True,
                 strategy: Optional[Strategy] = None):
        self.num_items = num_items
        self.batch_size = batch_size
        self.should_shuffle = shuffle
        self.strategy = strategy or Strategy()
        self.indexes = list(self.strategy.range(0, self.num_items))
        self.on_epoch_end()

    def __len__(self) -> int:
        return math.ceil(self.num_items / self.batch_size)

    def __getitem__(self, index: int) -> tuple[tuple[int, int], list[int]]:
        low = self.batch_size * index
        # Cap upper bound at array length; the last batch may be smaller
        # if the total number of items is not a multiple of batch size.
        high = min(low + self.batch_size, self.num_items)
        positions = self.indexes[low:high]
        return (low, high), positions

    def on_epoch_end(self):
        if self.should_shuffle:
            self.strategy.shuffle(self.indexes)
