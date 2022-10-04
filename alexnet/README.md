# AlexNet

In this directory, we aim to implement the AlexNet architecture for a
Convolutional Neural Network (CNN) used for image classification, to be tested
with the ImageNet dataset.

## CIFAR-10 implementations

| Description | Library | Notebook |
|:-----------:|:-------:|:--------:|
| Using Pylearn2/Keras LRN | Keras | [![View on GitHub][github-badge]][github-cifar10-pylearn2-lrn] [![Open In Colab][colab-badge]][colab-cifar10-pylearn2-lrn] [![Open in Binder][binder-badge]][binder-cifar10-pylearn2-lrn] |
| Using TF.NN.LRN | Keras | [![View on GitHub][github-badge]][github-cifar10-tf-nn-lrn] [![Open In Colab][colab-badge]][colab-cifar10-tf-nn-lrn] [![Open in Binder][binder-badge]][binder-cifar10-tf-nn-lrn] |

## ImageNet implementations

|      | Description    | Library | Notebook |
|:----:| -------------- |:-------:|:--------:|
|  v1  | Basic impl     |  Keras  | [![View on GitHub][github-badge]][github-basic] [![Open In Colab][colab-badge]][colab-basic] [![Open in Binder][binder-badge]][binder-basic] |

Implementation notes for ImageNet v1:

1. We haven't yet trained or tested this network, as we don't yet have access to
   the ImageNet dataset which requires registration & approval to be able to
   download it.
2. We haven't yet implemented the Local Response Normalization layer after the
   first two convolutional passes; the points where they should be added are
   marked with `TODO`s in the notebook.
3. This simple implementation does not split the data across 2 GPUs as described
   in the paper for simplicity, and because we no longer have such resource
   constraints in today's GPUs.

## References

Our implementation is based on the following paper:

* Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. 2012. ImageNet
  Classification with Deep Convolutional Neural Networks. In Proceedings of the
  25th International Conference on Neural Information Processing Systems -
  Volume 1 (NeurIPS 2012). Curran Associates Inc., Red Hook, NY, USA, 1097–1105.

The paper is available via:

* [NeurIPS][neurips-alexnet]
* [ACM Digital Library][acm-alexnet]
* [CiteSeerX][citeseer-alexnet]

See also:

* [Slides on ImageNet][imagenet-slides]
* [AlexNet on Wikipedia][alexnet-wiki]

## Local Response Normalization

Per the AlexNet paper, the _Local Response Normalization_ layer computes the
following function:

$$
b_{x,y}^i = a_{x,y}^i /
    \left(
        k + \alpha \sum_{j = \max(0, i-n/2)}^{\min(N-1, i+n/2)}
        \left(a_{x,y}^j \right)^2
    \right) ^ \beta
$$

The paper authors chose $k = 2, n = 5, \alpha = 10^{-4}, \beta = 0.75$.

We provide 2 implementations of the Local Response Normalization layer:

* one in this directory,
  [`local_response_normalization.py`](local_response_normaliation.py) as a very
  light wrapper around [`tensorflow.nn.local_response_normalization`][tf-nn-lrn]
  layer which was written as a result of this paper
* another one in
  [`third_party/pylearn2/local_response_normalization.py`](../third_party/pylearn2/local_response_normalization.py)
  which is based on the Pylearn2 implementation and adapted in the Keras project

## Input size discrepancy

Note that the original AlexNet paper refers to inputs as $224 \times 224$
images; however, that does not work out to have $55 \times 55$ images as the
output from the first convolutional layer. There's consensus that the only way
to achieve that is to use input images of size $227 \times 227$, or images of
size $224 \times 224$ with a 3-pixel zero-padding, which makes it work.

Here are several references which agree on this analysis of input shape:

1. [Classic Networks][classic-networks] lecture by Andrew Ng as part of the Deep
   Learning specialization

   > If you read the paper, the paper refers to $224 \times 224 \times 3$
   > images, but if you look at the numbers, the numbers only make sense if they
   > are $227 \times 227 \times 3$.

2. [Stanford CS231n][stanford-cs231n]

   > As a fun aside, if you read the actual paper it claims that the input
   > images were 224x224, which is surely incorrect because (224 - 11)/4 + 1 is
   > quite clearly not an integer. This has confused many people in the history
   > of ConvNets and little is known about what happened. My own best guess is
   > that Alex used zero-padding of 3 extra pixels that he does not mention in
   > the paper.

3. [Data Science SE][datascience-se]

   One answer also quotes Andrew Ng's lecture in (1) above. Another answer
   demonstrates via calculation that the $224 \times 224$ input shape would
   result in a non-integral output shape, and hence, must be an error, similarly
   to the Stanford CS231n class notes in (2) above.

[github-badge]: https://img.shields.io/badge/View-on%20GitHub-blue?logo=GitHub
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[binder-badge]: https://static.mybinder.org/badge_logo.svg

[github-cifar10-pylearn2-lrn]: AlexNet_for_CIFAR-10_with_Pylearn2_Keras_LRN.ipynb
[colab-cifar10-pylearn2-lrn]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/alexnet/AlexNet_for_CIFAR-10_with_Pylearn2_Keras_LRN.ipynb
[binder-cifar10-pylearn2-lrn]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=alexnet/AlexNet_for_CIFAR-10_with_Pylearn2_Keras_LRN.ipynb

[github-cifar10-tf-nn-lrn]: AlexNet_for_CIFAR-10_in_Keras_with_tf_nn_LocalResponseNormalization.ipynb
[colab-cifar10-tf-nn-lrn]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/alexnet/AlexNet_for_CIFAR-10_in_Keras_with_tf_nn_LocalResponseNormalization.ipynb
[binder-cifar10-tf-nn-lrn]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=alexnet/AlexNet_for_CIFAR-10_in_Keras_with_tf_nn_LocalResponseNormalization.ipynb

[github-basic]: Basic_AlexNet_in_Keras.ipynb
[colab-basic]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/alexnet/Basic_AlexNet_in_Keras.ipynb
[binder-basic]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=alexnet/Basic_AlexNet_in_Keras.ipynb

[neurips-alexnet]: https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
[acm-alexnet]: https://dl.acm.org/doi/10.5555/2999134.2999257
[citeseer-alexnet]: http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.299.205

[imagenet-slides]: https://image-net.org/static_files/files/supervision.pdf
[alexnet-wiki]: https://en.wikipedia.org/wiki/AlexNet

[classic-networks]: https://youtu.be/dZVkygnKh1M?t=421
[stanford-cs231n]: https://cs231n.github.io/convolutional-networks/
[datascience-se]: https://datascience.stackexchange.com/questions/29245/what-is-the-input-size-of-alex-net

[tf-nn-lrn]: https://www.tensorflow.org/api_docs/python/tf/nn/local_response_normalization
