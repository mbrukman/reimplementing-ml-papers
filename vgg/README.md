# VGG

In this directory, we aim to implement the VGG family of convolutional neural
network (CNN) models for image classification, including the well-known VGG-16
and VGG-19 models, to be tested with the ImageNet dataset.

Available implementations:

|      | Description    | Library | GitHub<br/>(readonly) | Colab | Binder |
|:----:| -------------- |:-------:|:---------------------:|:-----:|:------:|
|  v1  | Basic impl     |  Keras  | [![View on GitHub][github-badge]][github-basic] | [![Open In Colab][colab-badge]][colab-basic] | [![Open in Binder][binder-badge]][binder-basic] |

Implementation notes for v1:

1. We haven't yet trained or tested this network, as we don't yet have access to
   the ImageNet dataset which requires registration & approval to be able to
   download it.
2. We haven't yet implemented the Local Response Normalization layer (borrowed
   from AlexNet), but the authors of the VGG paper noted that it did not provide
   an improvement over not using it.

Our implementation is based on the following paper:

* Karen Simonyan and Andrew Zisserman. "Very deep convolutional networks for
  large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).

> Note: this paper was also published in ICLR 2015.

This paper is available via:

* [Visual Geometry Group at Oxford][paper-vgg]
* [arXiv][arxiv-vgg]
* [CiteSeerX][citeseerx-vgg]

See also:

* [VGG report from the authors][model-info] - links to download models
* [Papers with Code][pwc-vgg]

[github-badge]: https://img.shields.io/badge/View-GitHub-blue
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[binder-badge]: https://static.mybinder.org/badge_logo.svg

[github-basic]: Basic_VGG_in_Keras.ipynb
[colab-basic]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/vgg/Basic_VGG_in_Keras.ipynb
[binder-basic]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=vgg/Basic_VGG_in_Keras.ipynb

[paper-vgg]: https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/
[arxiv-vgg]: https://arxiv.org/abs/1409.1556
[citeseerx-vgg]: https://citeseerx.ist.psu.edu/viewdoc/summary;?doi=10.1.1.740.6937
[model-info]: https://www.robots.ox.ac.uk/~vgg/research/very_deep/
[pwc-vgg]: https://paperswithcode.com/method/vgg
