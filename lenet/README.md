# LeNet

This directory aims to implement the LeNet architecture for a Convolutional
Neural Network (CNN) used for character recognition, trained and tested with the
[MNIST dataset][mnist-tf].

Available implementations:

|     | Description    | Input<br/>pixel<br/>range | Pooling<br/>layer | Activation | Learning<br/>rate | Library | Notebook |
|:---:| -------------- |:-------------------------:|:-----------------:|:----------:|:-----------------:|:-------:|:--------:|
| v1 | Basic impl | [0, 1] | MaxPool2D | tanh | 0.001 | Keras | [![View on GitHub][github-badge]][github-keras-v1] [![Open In Colab][colab-badge]][colab-keras-v1] [![Open in Binder][binder-badge]][binder-keras-v1] |
| v2 | Custom layer,<br/>activation | [0, 1] | Subsampling | scaled<br/>tanh | 0.001 | Keras | [![View on GitHub][github-badge]][github-keras-v2] [![Open In Colab][colab-badge]][colab-keras-v2] [![Open in Binder][binder-badge]][binder-keras-v2] |
| v3 | Custom layer,<br/>activation,<br/>scaling,<br/>learning rate | [-0.1, 1.175] | Subsampling | scaled<br/>tanh | schedule | Keras | [![View on GitHub][github-badge]][github-keras-v3] [![Open In Colab][colab-badge]][colab-keras-v3] [![Open in Binder][binder-badge]][binder-keras-v3] |

Our implementation is based on the following paper:

* Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based
  learning applied to document recognition. _Proceedings of the IEEE,_
  86(11):2278-2324, November 1998.

The paper is available via:

* [Yann LeCun's website](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
* [Stanford CS 598](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
* [CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.138.1115)
* [Google Scholar][lenet-google-scholar]

See also:

* LeNet on [Wikipedia][lenet-wikipedia]
* MNIST on [Wikipedia][mnist-wikipedia], [Hugging Face][mnist-hf], [Papers with Code][mnist-pwc]

[github-badge]: https://img.shields.io/badge/View-on%20GitHub-blue?logo=GitHub
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[binder-badge]: https://static.mybinder.org/badge_logo.svg

[github-keras-v1]: LeNet_v1_basic_impl_in_Keras.ipynb
[colab-keras-v1]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/lenet/LeNet_v1_basic_impl_in_Keras.ipynb
[binder-keras-v1]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=lenet/LeNet_v1_basic_impl_in_Keras.ipynb

[github-keras-v2]: LeNet_v2_custom_Subsampling_layer_and_activation_in_Keras.ipynb
[colab-keras-v2]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/lenet/LeNet_v2_custom_Subsampling_layer_and_activation_in_Keras.ipynb
[binder-keras-v2]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=lenet/LeNet_v2_custom_Subsampling_layer_and_activation_in_Keras.ipynb

[github-keras-v3]: LeNet_v3_Subsamping_fixed_scaling_and_learning_rate_decay_in_Keras.ipynb
[colab-keras-v3]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/lenet/LeNet_v3_Subsamping_fixed_scaling_and_learning_rate_decay_in_Keras.ipynb
[binder-keras-v3]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=lenet/LeNet_v3_Subsamping_fixed_scaling_and_learning_rate_decay_in_Keras.ipynb

[lenet-google-scholar]: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=WLN3QrAAAAAJ&citation_for_view=WLN3QrAAAAAJ:u5HHmVD_uO8C
[lenet-wikipedia]: https://en.wikipedia.org/wiki/LeNet

[mnist-hf]: https://huggingface.co/datasets/mnist
[mnist-pwc]: https://paperswithcode.com/dataset/mnist
[mnist-tf]: https://www.tensorflow.org/datasets/catalog/mnist
[mnist-wikipedia]: https://en.wikipedia.org/wiki/MNIST_database
