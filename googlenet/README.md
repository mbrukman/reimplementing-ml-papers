# GoogLeNet

In this directory, we aim to implement the GoogLeNet convolutional neural
network (CNN) model for image classification, to be tested with the ImageNet
dataset.

In the [`diagrams`](diagrams) subdirectory, we recreated the GoogLeNet
architecture diagram in SVG, as well as an annotated version of the architecture
with the Inception modules highlighted, and a simplified version with each
Inception module collapsed to a single node, which makes it easier to visualize
and implement the network. See the directory's documentation for details.

Available implementations:

|      | Description    | Library | Notebook |
|:----:| -------------- |:-------:|:--------:|
|  v1  | Basic impl     |  Keras  | [![View on GitHub][github-badge]][github-basic] [![Open In Colab][colab-badge]][colab-basic] [![Open in Binder][binder-badge]][binder-basic] |

Implementation notes for v1:

1. We haven't yet trained or tested this network, as we don't yet have access to
   the ImageNet dataset which requires registration & approval to be able to
   download it.

Our implementation is based on the following paper:

* Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott E. Reed,
  Dragomir Anguelov, D. Erhan, Vincent Vanhoucke, and Andrew Rabinovich. “Going
  deeper with convolutions.” _2015 IEEE Conference on Computer Vision and
  Pattern Recognition (CVPR)_ (2015): 1-9.

You can access this paper via:

* [arXiv][arxiv-googlenet]
* [Google Scholar][scholar-googlenet]

See also:

* [Papers with Code][pwc-googlenet]

[github-badge]: https://img.shields.io/badge/View-on%20GitHub-blue?logo=GitHub
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[binder-badge]: https://static.mybinder.org/badge_logo.svg

[github-basic]: GoogLeNet_implementation_in_Keras.ipynb
[colab-basic]: https://colab.research.google.com/github/mbrukman/reimplementing-ml-papers/blob/main/googlenet/GoogLeNet_implementation_in_Keras.ipynb
[binder-basic]: https://mybinder.org/v2/gh/mbrukman/reimplementing-ml-papers/main?filepath=googlenet/GoogLeNet_implementation_in_Keras.ipynb

[arxiv-googlenet]: https://arxiv.org/abs/1409.4842
[scholar-googlenet]: https://scholar.google.com/scholar_lookup?arxiv_id=1409.4842
[pwc-googlenet]: https://paperswithcode.com/paper/going-deeper-with-convolutions
