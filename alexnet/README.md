# AlexNet

In this directory, we aim to implement the AlexNet architecture for a
Convolutional Neural Network (CNN) used for image classification, to be tested
with the ImageNet dataset.

You can see an implementation in a [notebook](Basic_AlexNet_in_Keras.ipynb).

> Note: we haven't yet trained or tested this network, as we don't yet have
> access to the ImageNet dataset which requires registration & approval to be
> able to download it.

Our implementation is based on the following paper:

* Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. 2012. ImageNet
  Classification with Deep Convolutional Neural Networks. In Proceedings of the
  25th International Conference on Neural Information Processing Systems -
  Volume 1 (NeurIPS 2012). Curran Associates Inc., Red Hook, NY, USA, 1097–1105.

The paper is available via:

* [NeurIPS](https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html)
* [ACM Digital Library](https://dl.acm.org/doi/10.5555/2999134.2999257)
* [CiteSeerX](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.299.205)

See also:

* [Slides on ImageNet](https://image-net.org/static_files/files/supervision.pdf)
* [AlexNet on Wikipedia](https://en.wikipedia.org/wiki/AlexNet)

# Differences from the paper

This simple implementation does not split the data across 2 GPUs as described in
the original paper for simplicity of implementation.

Additionally, we haven't yet implemented the Local Response Normalization after
the first two convolutional passes; the points where they should be added are
marked with `TODO`s in the notebook.

# Input size discrepancy

Note that the original AlexNet paper refers to inputs as $224 \times 224$
images; however, that does not work out to have $55 \times 55$ images as the
output from the first convolutional layer. There's consensus that the only way
to achieve that is to use input images of size $227 \times 227$, or images of
size $224 \times 224$ with a 3-pixel zero-padding, which makes it work.

Here are several references which agree on this analysis of input shape:

1. [Classic Networks](https://youtu.be/dZVkygnKh1M?t=421) lecture by Andrew Ng
   as part of the Deep Learning specialization

   > If you read the paper, the paper refers to $224 \times 224 \times 3$
   > images, but if you look at the numbers, the numbers only make sense if they
   > are $227 \times 227 \times 3$.

2. [Stanford CS231n](https://cs231n.github.io/convolutional-networks/)

   > As a fun aside, if you read the actual paper it claims that the input
   > images were 224x224, which is surely incorrect because (224 - 11)/4 + 1 is
   > quite clearly not an integer. This has confused many people in the history
   > of ConvNets and little is known about what happened. My own best guess is
   > that Alex used zero-padding of 3 extra pixels that he does not mention in
   > the paper.

3. [Data Science SE](https://datascience.stackexchange.com/questions/29245/what-is-the-input-size-of-alex-net)

   One answer also quotes Andrew Ng's lecture in (1) above. Another answer
   demonstrates via calculation that the $224 \times 224$ input shape would
   result in a non-integral output shape, and hence, must be an error, similarly
   to the Stanford CS231n class notes in (2) above.
