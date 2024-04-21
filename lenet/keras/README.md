# LeNet in Keras

See [implementation notes](notes.md) which includes explanation of the various
implementation versions, and additional details, such as an extension of the
custom Subsampling layer.

| Approach | Description    | Input<br/>pixel<br/>range | Pooling<br/>layer | Activation | Learning<br/>rate | Notebook |
|:--------:|:--------------:|:-------------------------:|:-----------------:|:----------:|:-----------------:|:--------:|
| v1 | Basic implementation | [0, 1] | MaxPool2D | tanh | 0.001 | [![View on GitHub][github-badge]][github-keras-v1] [![Open In Colab][colab-badge]][colab-keras-v1] [![Open in Binder][binder-badge]][binder-keras-v1] |
| v2 | Custom layer,<br/>activation | [0, 1] | Subsampling | scaled<br/>tanh | 0.001 | [![View on GitHub][github-badge]][github-keras-v2] [![Open In Colab][colab-badge]][colab-keras-v2] [![Open in Binder][binder-badge]][binder-keras-v2] |
| v3 | Custom layer,<br/>activation,<br/>scaling,<br/>learning rate | [-0.1, 1.175] | Subsampling | scaled<br/>tanh | schedule | [![View on GitHub][github-badge]][github-keras-v3] [![Open In Colab][colab-badge]][colab-keras-v3] [![Open in Binder][binder-badge]][binder-keras-v3] |

[github-badge]: https://img.shields.io/badge/View-on%20GitHub-blue?logo=GitHub
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[binder-badge]: https://static.mybinder.org/badge_logo.svg

[github-keras-v1]: LeNet_Keras_v1_basic_implementation.ipynb
[colab-keras-v1]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/lenet/keras/LeNet_Keras_v1_basic_implementation.ipynb
[binder-keras-v1]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=lenet/keras/LeNet_Keras_v1_basic_implementation.ipynb

[github-keras-v2]: LeNet_Keras_v2_custom_Subsampling_layer_and_activation.ipynb
[colab-keras-v2]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/lenet/keras/LeNet_Keras_v2_custom_Subsampling_layer_and_activation.ipynb
[binder-keras-v2]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=lenet/keras/LeNet_Keras_v2_custom_Subsampling_layer_and_activation.ipynb

[github-keras-v3]: LeNet_Keras_v3_Subsamping_fixed_scaling_and_learning_rate_decay.ipynb
[colab-keras-v3]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/lenet/keras/LeNet_Keras_v3_Subsamping_fixed_scaling_and_learning_rate_decay.ipynb
[binder-keras-v3]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=lenet/keras/LeNet_Keras_v3_Subsamping_fixed_scaling_and_learning_rate_decay.ipynb
