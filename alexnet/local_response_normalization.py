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
from keras.layers import Layer


class LocalResponseNormalization(Layer):
    bias: float
    depth_radius: int
    alpha: float
    beta: float

    def __init__(self, k=2, n=5, alpha=1e-4, beta=0.75, **kwargs):
        super().__init__(**kwargs)
        self.bias = k
        self.depth_radius = n
        self.alpha = alpha
        self.beta = beta

    def call(self, input_):
        # Interestingly enough, the documentation for this function:
        # https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
        # actually cites the AlexNet paper for the implementation.
        return tf.nn.local_response_normalization(
            input_, self.depth_radius, self.bias, self.alpha, self.beta)
