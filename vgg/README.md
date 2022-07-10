# VGG

In this directory, we aim to implement the VGG family of convolutional neural
network (CNN) models for image classification, including the well-known VGG-16
and VGG-19 models, to be tested with the ImageNet dataset.

You can see our implementation in a [notebook](Basic_VGG_in_Keras.ipynb).

Implementation notes:

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

[paper-vgg]: https://www.robots.ox.ac.uk/~vgg/publications/2015/Simonyan15/
[arxiv-vgg]: https://arxiv.org/abs/1409.1556
[citeseerx-vgg]: https://citeseerx.ist.psu.edu/viewdoc/summary;?doi=10.1.1.740.6937
[model-info]: https://www.robots.ox.ac.uk/~vgg/research/very_deep/
[pwc-vgg]: https://paperswithcode.com/method/vgg
