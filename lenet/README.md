# LeNet

This directory aims to implement the LeNet architecture for a Convolutional
Neural Network (CNN) used for character recognition, and tested with the MNIST
dataset.

Available implementations:

| Implementation | Library | GitHub preview<br/>(readonly) | Colab | Binder |
| -------------- | ------- | ----------------------------- | ----- | ------ |
| Basic impl using MaxPool2D and ReLU | Keras | [GitHub][github-basic] | [![Open In Colab][colab-badge]][colab-basic] | [![Open in Binder][binder-badge]][binder-basic] |
| Subsampling layer + LeNet activation | Keras | [GitHub][github-subsampling] | [![Open In Colab][colab-badge]][colab-subsampling] | [![Open in Binder][binder-badge]][binder-subsampling] |

Our implementation is based on the following paper:

Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. Gradient-based learning applied
to document recognition. _Proceedings of the IEEE,_ 86(11):2278-2324, November
1998.

The paper is available via:

* [Yann LeCun's website](http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf)
* [Stanford CS 598](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)
* [CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.138.1115)
* [Google Scholar][lenet-google-scholar]

See also:

* [LeNet on Wikipedia](https://en.wikipedia.org/wiki/LeNet)

[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[binder-badge]: https://static.mybinder.org/badge_logo.svg

[github-basic]: Basic_LeNet_in_Keras.ipynb
[colab-basic]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/lenet/Basic_LeNet_in_Keras.ipynb
[binder-basic]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=lenet/Basic_LeNet_in_Keras.ipynb

[github-subsampling]: LeNet_with_Subsampling_in_Keras.ipynb
[colab-subsampling]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/lenet/LeNet_with_Subsampling_in_Keras.ipynb
[binder-subsampling]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=lenet/LeNet_with_Subsampling_in_Keras.ipynb

[lenet-google-scholar]: https://scholar.google.com/citations?view_op=view_citation&hl=en&user=WLN3QrAAAAAJ&citation_for_view=WLN3QrAAAAAJ:u5HHmVD_uO8C
