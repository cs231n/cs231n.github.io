{
 "nbformat_minor": 0, 
 "nbformat": 4, 
 "cells": [
  {
   "source": [
    "# Image Gradients\n", 
    "In this notebook we'll introduce the TinyImageNet dataset and a deep CNN that has been pretrained on this dataset. You will use this pretrained model to compute gradients with respect to images, and use these image gradients to produce class saliency maps and fooling images."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# As usual, a bit of setup\n", 
    "\n", 
    "import time, os, json\n", 
    "import numpy as np\n", 
    "import skimage.io\n", 
    "import matplotlib.pyplot as plt\n", 
    "\n", 
    "from cs231n.classifiers.pretrained_cnn import PretrainedCNN\n", 
    "from cs231n.data_utils import load_tiny_imagenet\n", 
    "from cs231n.image_utils import blur_image, deprocess_image\n", 
    "\n", 
    "%matplotlib inline\n", 
    "plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots\n", 
    "plt.rcParams['image.interpolation'] = 'nearest'\n", 
    "plt.rcParams['image.cmap'] = 'gray'\n", 
    "\n", 
    "# for auto-reloading external modules\n", 
    "# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython\n", 
    "%load_ext autoreload\n", 
    "%autoreload 2"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Introducing TinyImageNet\n", 
    "\n", 
    "The TinyImageNet dataset is a subset of the ILSVRC-2012 classification dataset. It consists of 200 object classes, and for each object class it provides 500 training images, 50 validation images, and 50 test images. All images have been downsampled to 64x64 pixels. We have provided the labels for all training and validation images, but have withheld the labels for the test images.\n", 
    "\n", 
    "We have further split the full TinyImageNet dataset into two equal pieces, each with 100 object classes. We refer to these datasets as TinyImageNet-100-A and TinyImageNet-100-B; for this exercise you will work with TinyImageNet-100-A.\n", 
    "\n", 
    "To download the data, go into the `cs231n/datasets` directory and run the script `get_tiny_imagenet_a.sh`. Then run the following code to load the TinyImageNet-100-A dataset into memory.\n", 
    "\n", 
    "NOTE: The full TinyImageNet-100-A dataset will take up about 250MB of disk space, and loading the full TinyImageNet-100-A dataset into memory will use about 2.8GB of memory."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "data = load_tiny_imagenet('cs231n/datasets/tiny-imagenet-100-A', subtract_mean=True)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# TinyImageNet-100-A classes\n", 
    "Since ImageNet is based on the WordNet ontology, each class in ImageNet (and TinyImageNet) actually has several different names. For example \"pop bottle\" and \"soda bottle\" are both valid names for the same class. Run the following to see a list of all classes in TinyImageNet-100-A:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "for i, names in enumerate(data['class_names']):\n", 
    "  print i, ' '.join('\"%s\"' % name for name in names)"
   ], 
   "outputs": [], 
   "metadata": {
    "scrolled": false, 
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Visualize Examples\n", 
    "Run the following to visualize some example images from random classses in TinyImageNet-100-A. It selects classes and images randomly, so you can run it several times to see different images."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Visualize some examples of the training data\n", 
    "classes_to_show = 7\n", 
    "examples_per_class = 5\n", 
    "\n", 
    "class_idxs = np.random.choice(len(data['class_names']), size=classes_to_show, replace=False)\n", 
    "for i, class_idx in enumerate(class_idxs):\n", 
    "  train_idxs, = np.nonzero(data['y_train'] == class_idx)\n", 
    "  train_idxs = np.random.choice(train_idxs, size=examples_per_class, replace=False)\n", 
    "  for j, train_idx in enumerate(train_idxs):\n", 
    "    img = deprocess_image(data['X_train'][train_idx], data['mean_image'])\n", 
    "    plt.subplot(examples_per_class, classes_to_show, 1 + i + classes_to_show * j)\n", 
    "    if j == 0:\n", 
    "      plt.title(data['class_names'][class_idx][0])\n", 
    "    plt.imshow(img)\n", 
    "    plt.gca().axis('off')\n", 
    "\n", 
    "plt.show()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Pretrained model\n", 
    "We have trained a deep CNN for you on the TinyImageNet-100-A dataset that we will use for image visualization. The model has 9 convolutional layers (with spatial batch normalization) and 1 fully-connected hidden layer (with batch normalization).\n", 
    "\n", 
    "To get the model, run the script `get_pretrained_model.sh` from the `cs231n/datasets` directory. After doing so, run the following to load the model from disk."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "model = PretrainedCNN(h5_file='cs231n/datasets/pretrained_model.h5')"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "## Pretrained model performance\n", 
    "Run the following to test the performance of the pretrained model on some random training and validation set images. You should see training accuracy around 90% and validation accuracy around 60%; this indicates a bit of overfitting, but it should work for our visualization experiments."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "batch_size = 100\n", 
    "\n", 
    "# Test the model on training data\n", 
    "mask = np.random.randint(data['X_train'].shape[0], size=batch_size)\n", 
    "X, y = data['X_train'][mask], data['y_train'][mask]\n", 
    "y_pred = model.loss(X).argmax(axis=1)\n", 
    "print 'Training accuracy: ', (y_pred == y).mean()\n", 
    "\n", 
    "# Test the model on validation data\n", 
    "mask = np.random.randint(data['X_val'].shape[0], size=batch_size)\n", 
    "X, y = data['X_val'][mask], data['y_val'][mask]\n", 
    "y_pred = model.loss(X).argmax(axis=1)\n", 
    "print 'Validation accuracy: ', (y_pred == y).mean()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Saliency Maps\n", 
    "Using this pretrained model, we will compute class saliency maps as described in Section 3.1 of [1].\n", 
    "\n", 
    "As mentioned in Section 2 of the paper, you should compute the gradient of the image with respect to the unnormalized class score, not with respect to the normalized class probability.\n", 
    "\n", 
    "You will need to use the `forward` and `backward` methods of the `PretrainedCNN` class to compute gradients with respect to the image. Open the file `cs231n/classifiers/pretrained_cnn.py` and read the documentation for these methods to make sure you know how they work. For example usage, you can see the `loss` method. Make sure to run the model in `test` mode when computing saliency maps.\n", 
    "\n", 
    "[1] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. \"Deep Inside Convolutional Networks: Visualising\n", 
    "Image Classification Models and Saliency Maps\", ICLR Workshop 2014."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "def compute_saliency_maps(X, y, model):\n", 
    "  \"\"\"\n", 
    "  Compute a class saliency map using the model for images X and labels y.\n", 
    "  \n", 
    "  Input:\n", 
    "  - X: Input images, of shape (N, 3, H, W)\n", 
    "  - y: Labels for X, of shape (N,)\n", 
    "  - model: A PretrainedCNN that will be used to compute the saliency map.\n", 
    "  \n", 
    "  Returns:\n", 
    "  - saliency: An array of shape (N, H, W) giving the saliency maps for the input\n", 
    "    images.\n", 
    "  \"\"\"\n", 
    "  saliency = None\n", 
    "  ##############################################################################\n", 
    "  # TODO: Implement this function. You should use the forward and backward     #\n", 
    "  # methods of the PretrainedCNN class, and compute gradients with respect to  #\n", 
    "  # the unnormalized class score of the ground-truth classes in y.             #\n", 
    "  ##############################################################################\n", 
    "  pass\n", 
    "  ##############################################################################\n", 
    "  #                             END OF YOUR CODE                               #\n", 
    "  ##############################################################################\n", 
    "  return saliency"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": true
   }
  }, 
  {
   "source": [
    "Once you have completed the implementation in the cell above, run the following to visualize some class saliency maps on the validation set of TinyImageNet-100-A."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "def show_saliency_maps(mask):\n", 
    "  mask = np.asarray(mask)\n", 
    "  X = data['X_val'][mask]\n", 
    "  y = data['y_val'][mask]\n", 
    "\n", 
    "  saliency = compute_saliency_maps(X, y, model)\n", 
    "\n", 
    "  for i in xrange(mask.size):\n", 
    "    plt.subplot(2, mask.size, i + 1)\n", 
    "    plt.imshow(deprocess_image(X[i], data['mean_image']))\n", 
    "    plt.axis('off')\n", 
    "    plt.title(data['class_names'][y[i]][0])\n", 
    "    plt.subplot(2, mask.size, mask.size + i + 1)\n", 
    "    plt.title(mask[i])\n", 
    "    plt.imshow(saliency[i])\n", 
    "    plt.axis('off')\n", 
    "  plt.gcf().set_size_inches(10, 4)\n", 
    "  plt.show()\n", 
    "\n", 
    "# Show some random images\n", 
    "mask = np.random.randint(data['X_val'].shape[0], size=5)\n", 
    "show_saliency_maps(mask)\n", 
    "  \n", 
    "# These are some cherry-picked images that should give good results\n", 
    "show_saliency_maps([128, 3225, 2417, 1640, 4619])"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Fooling Images\n", 
    "We can also use image gradients to generate \"fooling images\" as discussed in [2]. Given an image and a target class, we can perform gradient ascent over the image to maximize the target class, stopping when the network classifies the image as the target class. Implement the following function to generate fooling images.\n", 
    "\n", 
    "[2] Szegedy et al, \"Intriguing properties of neural networks\", ICLR 2014"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "def make_fooling_image(X, target_y, model):\n", 
    "  \"\"\"\n", 
    "  Generate a fooling image that is close to X, but that the model classifies\n", 
    "  as target_y.\n", 
    "  \n", 
    "  Inputs:\n", 
    "  - X: Input image, of shape (1, 3, 64, 64)\n", 
    "  - target_y: An integer in the range [0, 100)\n", 
    "  - model: A PretrainedCNN\n", 
    "  \n", 
    "  Returns:\n", 
    "  - X_fooling: An image that is close to X, but that is classifed as target_y\n", 
    "    by the model.\n", 
    "  \"\"\"\n", 
    "  X_fooling = X.copy()\n", 
    "  ##############################################################################\n", 
    "  # TODO: Generate a fooling image X_fooling that the model will classify as   #\n", 
    "  # the class target_y. Use gradient ascent on the target class score, using   #\n", 
    "  # the model.forward method to compute scores and the model.backward method   #\n", 
    "  # to compute image gradients.                                                #\n", 
    "  #                                                                            #\n", 
    "  # HINT: For most examples, you should be able to generate a fooling image    #\n", 
    "  # in fewer than 100 iterations of gradient ascent.                           #\n", 
    "  ##############################################################################\n", 
    "  pass\n", 
    "  ##############################################################################\n", 
    "  #                             END OF YOUR CODE                               #\n", 
    "  ##############################################################################\n", 
    "  return X_fooling"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": true
   }
  }, 
  {
   "source": [
    "Run the following to choose a random validation set image that is correctly classified by the network, and then make a fooling image."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Find a correctly classified validation image\n", 
    "while True:\n", 
    "  i = np.random.randint(data['X_val'].shape[0])\n", 
    "  X = data['X_val'][i:i+1]\n", 
    "  y = data['y_val'][i:i+1]\n", 
    "  y_pred = model.loss(X)[0].argmax()\n", 
    "  if y_pred == y: break\n", 
    "\n", 
    "target_y = 67\n", 
    "X_fooling = make_fooling_image(X, target_y, model)\n", 
    "\n", 
    "# Make sure that X_fooling is classified as y_target\n", 
    "scores = model.loss(X_fooling)\n", 
    "assert scores[0].argmax() == target_y, 'The network is not fooled!'\n", 
    "\n", 
    "# Show original image, fooling image, and difference\n", 
    "plt.subplot(1, 3, 1)\n", 
    "plt.imshow(deprocess_image(X, data['mean_image']))\n", 
    "plt.axis('off')\n", 
    "plt.title(data['class_names'][y][0])\n", 
    "plt.subplot(1, 3, 2)\n", 
    "plt.imshow(deprocess_image(X_fooling, data['mean_image'], renorm=True))\n", 
    "plt.title(data['class_names'][target_y][0])\n", 
    "plt.axis('off')\n", 
    "plt.subplot(1, 3, 3)\n", 
    "plt.title('Difference')\n", 
    "plt.imshow(deprocess_image(X - X_fooling, data['mean_image']))\n", 
    "plt.axis('off')\n", 
    "plt.show()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }
 ], 
 "metadata": {
  "kernelspec": {
   "display_name": "Python 2", 
   "name": "python2", 
   "language": "python"
  }, 
  "language_info": {
   "mimetype": "text/x-python", 
   "nbconvert_exporter": "python", 
   "name": "python", 
   "file_extension": ".py", 
   "version": "2.7.6", 
   "pygments_lexer": "ipython2", 
   "codemirror_mode": {
    "version": 2, 
    "name": "ipython"
   }
  }
 }
}