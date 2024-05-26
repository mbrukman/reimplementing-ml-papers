# LeNet in Keras implementation notes

Most LeNet implementations use the `AveragePooling2D` or `MaxPooling2D` layers
in place of the custom `Subsampling` layer which isn't a standard layer in Keras
or TensorFlow framework, so it typically looks as follows:

```python
from tensorflow import keras
from keras import Input, Sequential
from keras.layers import Activation, AveragePooling2D, Conv2D, Dense, Flatten


tanh = keras.activations.tanh
softmax = keras.activations.softmax

model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation=tanh, name='C1'),
    AveragePooling2D(pool_size=(2, 2), strides=(2, 2), name='S2'),
    Activation(tanh, name='S2_act'),
    Conv2D(filters=16, kernel_size=(5, 5), activation=tanh, name='C3'),
    subsampling(pool_size=(2, 2), strides=(2, 2), name='S4'),
    Activation(tanh, name='S4_act'),
    Conv2D(filters=120, kernel_size=(5, 5), activation=tanh, name='C5'),
    Flatten(name='Flatten'),
    Dense(84, activation=tanh, name='F6'),
    Dense(10, activation=softmax, name='Output'),
], name='LeNet-5')
```

Some folks end up using ReLU activation function instead of $\tanh$, some folks
use a `Dense` layer instead of `Conv2D` for the `C5` layer, etc., leading to a
number of variations.

One way to approach this is to create a generalized version of LeNet and allow
customization of the model, e.g., by providing a constructor function that lets
you have a custom subsampling layer and a custom activation function for
intermediate layers:

```python
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
```

> [!NOTE]
> The last layer **must** have a `softmax` activation to provide probabilities
> for the 10 output nodes.

With the above code in hand (also available in [`lenet.py`](lenet.py) in this
directory), we can now construct various versions of the model very easily:

```python
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.activations import relu

from lenet import LeNet # local import

# Standard models with AveragePooling2D or MaxPooling2D and tanh
model_avg = LeNet(subsampling=AveragePooling2D)
model_max = LeNet(subsampling=MaxPooling2D)

# Models with AveragePooling2D or MaxPooling2D and ReLU
reli = keras.activations.relu
model_avg = LeNet(subsampling=AveragePooling2D, activation=relu)
model_max = LeNet(subsampling=MaxPooling2D, activation=relu)
```

We can also try other activation functions, e.g., [sigmoid][sigmoid-fn],
[selu][selu-fn], [elu][elu-fn], or write our own custom one and provide it as a
parameter to `LeNet()`.

## Subsampling layer

Finally, we can also implement the [`Subsampling`](subsampling.py) layer as
described in the LeNet paper as well as the custom [scaled `tanh` activation
function](activations.py), and easily construct a LeNet model using these
parameters:

```python
# All local imports.
from activations import scaled_tanh
from lenet import LeNet
from subsampling import Subsampling


model = LeNet(subsampling=Subsampling, activation=scaled_tanh)
```

While implementing the Subsampling layer, we also implemented an extension which
can be found in [`subsampling_ext.py`](subsampling_ext.py): this version has a
(weight, bias) pair of parameters for each cell in the output, rather than just
a single pair of parameters (per channel) for the layer overall as described in
the LeNet paper.

Basic testing did not show a significant improvement in accuracy, but with any
increase in parameters, it does increase the training time.

## Implementation versions

We've tried to structure the v1, v2, v3, etc. notebooks as impleementations
which asymptotically approach the LeNet paper, each one implementing more
details in the LeNet paper than the previous. The table of versions in the
[`README.md`](README.md) shows the parameters that distinguish each
implementation from each other.

We've factored out common MNIST dataset processing details into a separate
library in [`../../datasets/mnist`](../../datasets/mnist) as well as the model
definition into [`lenet.py`](lenet.py) for easier reuse of common functionality
and to avoid code duplication. Thus, the notebooks are rather terse and not
entirely self-contained; if you want to see that version, it can be found in the
version control history of these files.


[sigmoid-fn]: https://keras.io/api/layers/activations/#sigmoid-function
[selu-fn]: https://keras.io/api/layers/activations/#selu-function
[elu-fn]: https://keras.io/api/layers/activations/#elu-function
