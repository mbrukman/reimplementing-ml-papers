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

* [Keras](keras)

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

[arxiv-googlenet]: https://arxiv.org/abs/1409.4842
[scholar-googlenet]: https://scholar.google.com/scholar_lookup?arxiv_id=1409.4842
[pwc-googlenet]: https://paperswithcode.com/paper/going-deeper-with-convolutions
