# Datasets

Available datasets:

* [CIFAR-10](cifar-10)
* [MNIST](mnist)

## Data loading in Keras via `Sequence` subclass

This directory also holds some sample code for implementing a subclass of a
[Keras `Sequence` class][keras-sequence] which can be used to provide data
during a model training or testing process.

See [`data_sequence.py`](data_sequence.py) for the code and
[`data_sequence_test.py`](data_sequence_test.py) for the tests.

These are provided since I've seen some sample code on various forums that are
incorrect in one way or another, and since they don't come with tests, they
require manual verification. Hopefully, this will help provide some clarity and
useful examples on which to base your `Sequence` implementations.

Why is this useful? There are a number of benefits:

* Per the [Keras documentation][keras-sequence]:

  > `Sequence` are a safer way to do multiprocessing. This structure guarantees
  > that the network will only train once on each sample per epoch which is not
  > the case with generators.

* You may not be able to load the entire train/validation/test dataset in memory
  at the same time, so you need a way to load it just in time for training and
  then load the next batch. This class makes it very easy to implement.

* You may want to do augmentation on your input dataset for training, but you
  don't want to generate it and write it out to disk, as it's much cheaper to do
  the data augmentation process in-memory than to read data from disk (and may
  require too much disk space to write out all the image versions).

* You can do various random augmentations dynamically in-memory at training
  time, rather than having to write out just a single version of the
  augmentation.

* The default Keras data augmentation options in functions such as
  [`image_dataset_from_directory`][keras-image_dataset_from_directory] are close
  to what you need, but don't provide all the options you would like, so you
  need to write your own version.


[keras-sequence]: https://keras.io/api/utils/python_utils/#sequence-class
[tf-keras-sequence]: https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
[tf-keras-image_dataset_from_directory]: https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
[keras-image_dataset_from_directory]: https://keras.io/api/data_loading/image/#imagedatasetfromdirectory-function

