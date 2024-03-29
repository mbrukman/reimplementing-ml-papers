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
"""Subsampling layer as defined in the LeNet paper."""

# From Python 3.9 and onward, `tuple`, `list` and other collection classes can
# also function as generic class types (see PEP 585).
#
# Once we no longer need to support Python 3.7 or 3.8, we can remove this syntax
# (added in PEP 563) for Python 3.7 and higher.
from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.layers import Layer


class ArgumentError(ValueError):
    pass


def identity(x: tf.Tensor) -> tf.Tensor:
    return x


class Subsampling(Layer):
    pool_size: tuple[int, int]
    strides: tuple[int, int]
    padding: str
    activation: Callable[[tf.Tensor], tf.Tensor]
    w: np.ndarray
    b: np.ndarray

    def __init__(
        self,
        pool_size: Union[int, list[int], tuple[int, int]] = (2, 2),
        strides: Optional[Union[int, list[int], tuple[int, int]]] = None,
        padding: str = 'VALID',
        activation: Callable[[tf.Tensor], tf.Tensor] = identity,
        **kwargs):
        """Subsampling layer as described in the LeNet paper.

        This layer is not a simple average or max pooling that are typically
        used to implement LeNet: neither average-pooling nor max-pooling have
        any trainable parameters, while the subsampling layer described in the
        LeNet paper *does* have trainable parameters.

        Note: assumes data format is `(batch_size, rows, cols, channels)`, i.e.,
        what TensorFlow / Keras describe as "channels_last".

        Args:
          pool_size: int or 2-tuple specifying pool size (aka kernel size)
          strides: int or 2-tuple; if unspecified, will be copied from
            `pool_size`
          padding:
        """
        super().__init__(**kwargs)

        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        elif (isinstance(pool_size, list) or
              isinstance(pool_size, tuple)) and len(pool_size) == 2:
            self.pool_size = (pool_size[0], pool_size[1])
        else:
            raise ArgumentError(
                f"`pool_size` must be an int or 2-tuple; received: {pool_size}")

        if strides is None:
            self.strides == self.pool_size
        elif isinstance(strides, int):
            self.strides = (strides, strides)
        elif (isinstance(strides, list) or
              isinstance(strides, tuple)) and len(strides) == 2:
            self.strides = (strides[0], strides[1])
        else:
            raise ArgumentError(
                f"`strides` must be an int or 2-tuple; received: {strides}")

        assert padding is not None and padding.upper() in ('VALID', 'SAME'), (
            f"`padding` must be either 'VALID' or 'SAME'; received: {padding}")
        self.padding = padding.upper()

        self.activation = activation

    def build(
        self, input_shape: tuple[Optional[int], int, int, int]) -> None:
        """Builds internal structures to prepare for model training.

        Args:
          input_shape: length-4 list or tuple representing (batch_size, rows,
            cols, channels); where `batch_size` may be None; see docs above for
            `__init__()` for details.
        """
        if len(input_shape) != 4:
            raise ArgumentError(
                f"`len(input_shape)` != 4; received: {input_shape}")
        if input_shape[0] is not None:
            raise ArgumentError(
                f"`input_shape[0] must be None; received: {input_shape}")

        in_chan = input_shape[3]

        self.w = self.add_weight(shape=(1, 1, in_chan),
                                 initializer="random_normal",
                                 trainable=True)

        self.b = self.add_weight(shape=(1, 1, in_chan),
                                 initializer="random_normal",
                                 trainable=True)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Computes subsampling value: `w * (sum of window entries) + b`."""

        # `scale` here undoes the average pooling by getting the original sum,
        # which is what we need, but there isn't a pooling mechanism that just
        # gets us the sum of products.
        scale = self.pool_size[0] * self.pool_size[1]
        tf_scale = tf.constant(scale, dtype='float32')

        avg = tf.nn.pool(inputs,
                         window_shape=self.pool_size,
                         pooling_type='AVG',
                         strides=self.strides,
                         padding=self.padding)

        return self.activation(self.w * tf_scale * avg + self.b)
