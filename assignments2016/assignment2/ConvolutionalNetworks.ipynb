{
 "nbformat_minor": 0, 
 "nbformat": 4, 
 "cells": [
  {
   "source": [
    "# Convolutional Networks\n", 
    "So far we have worked with deep fully-connected networks, using them to explore different optimization strategies and network architectures. Fully-connected networks are a good testbed for experimentation because they are very computationally efficient, but in practice all state-of-the-art results use convolutional networks instead.\n", 
    "\n", 
    "First you will implement several layer types that are used in convolutional networks. You will then use these layers to train a convolutional network on the CIFAR-10 dataset."
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
    "import numpy as np\n", 
    "import matplotlib.pyplot as plt\n", 
    "from cs231n.classifiers.cnn import *\n", 
    "from cs231n.data_utils import get_CIFAR10_data\n", 
    "from cs231n.gradient_check import eval_numerical_gradient_array, eval_numerical_gradient\n", 
    "from cs231n.layers import *\n", 
    "from cs231n.fast_layers import *\n", 
    "from cs231n.solver import Solver\n", 
    "\n", 
    "%matplotlib inline\n", 
    "plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots\n", 
    "plt.rcParams['image.interpolation'] = 'nearest'\n", 
    "plt.rcParams['image.cmap'] = 'gray'\n", 
    "\n", 
    "# for auto-reloading external modules\n", 
    "# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython\n", 
    "%load_ext autoreload\n", 
    "%autoreload 2\n", 
    "\n", 
    "def rel_error(x, y):\n", 
    "  \"\"\" returns relative error \"\"\"\n", 
    "  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Load the (preprocessed) CIFAR10 data.\n", 
    "\n", 
    "data = get_CIFAR10_data()\n", 
    "for k, v in data.iteritems():\n", 
    "  print '%s: ' % k, v.shape"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Convolution: Naive forward pass\n", 
    "The core of a convolutional network is the convolution operation. In the file `cs231n/layers.py`, implement the forward pass for the convolution layer in the function `conv_forward_naive`. \n", 
    "\n", 
    "You don't have to worry too much about efficiency at this point; just write the code in whatever way you find most clear.\n", 
    "\n", 
    "You can test your implementation by running the following:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "x_shape = (2, 3, 4, 4)\n", 
    "w_shape = (3, 3, 4, 4)\n", 
    "x = np.linspace(-0.1, 0.5, num=np.prod(x_shape)).reshape(x_shape)\n", 
    "w = np.linspace(-0.2, 0.3, num=np.prod(w_shape)).reshape(w_shape)\n", 
    "b = np.linspace(-0.1, 0.2, num=3)\n", 
    "\n", 
    "conv_param = {'stride': 2, 'pad': 1}\n", 
    "out, _ = conv_forward_naive(x, w, b, conv_param)\n", 
    "correct_out = np.array([[[[[-0.08759809, -0.10987781],\n", 
    "                           [-0.18387192, -0.2109216 ]],\n", 
    "                          [[ 0.21027089,  0.21661097],\n", 
    "                           [ 0.22847626,  0.23004637]],\n", 
    "                          [[ 0.50813986,  0.54309974],\n", 
    "                           [ 0.64082444,  0.67101435]]],\n", 
    "                         [[[-0.98053589, -1.03143541],\n", 
    "                           [-1.19128892, -1.24695841]],\n", 
    "                          [[ 0.69108355,  0.66880383],\n", 
    "                           [ 0.59480972,  0.56776003]],\n", 
    "                          [[ 2.36270298,  2.36904306],\n", 
    "                           [ 2.38090835,  2.38247847]]]]])\n", 
    "\n", 
    "# Compare your output to ours; difference should be around 1e-8\n", 
    "print 'Testing conv_forward_naive'\n", 
    "print 'difference: ', rel_error(out, correct_out)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Aside: Image processing via convolutions\n", 
    "\n", 
    "As fun way to both check your implementation and gain a better understanding of the type of operation that convolutional layers can perform, we will set up an input containing two images and manually set up filters that perform common image processing operations (grayscale conversion and edge detection). The convolution forward pass will apply these operations to each of the input images. We can then visualize the results as a sanity check."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "from scipy.misc import imread, imresize\n", 
    "\n", 
    "kitten, puppy = imread('kitten.jpg'), imread('puppy.jpg')\n", 
    "# kitten is wide, and puppy is already square\n", 
    "d = kitten.shape[1] - kitten.shape[0]\n", 
    "kitten_cropped = kitten[:, d/2:-d/2, :]\n", 
    "\n", 
    "img_size = 200   # Make this smaller if it runs too slow\n", 
    "x = np.zeros((2, 3, img_size, img_size))\n", 
    "x[0, :, :, :] = imresize(puppy, (img_size, img_size)).transpose((2, 0, 1))\n", 
    "x[1, :, :, :] = imresize(kitten_cropped, (img_size, img_size)).transpose((2, 0, 1))\n", 
    "\n", 
    "# Set up a convolutional weights holding 2 filters, each 3x3\n", 
    "w = np.zeros((2, 3, 3, 3))\n", 
    "\n", 
    "# The first filter converts the image to grayscale.\n", 
    "# Set up the red, green, and blue channels of the filter.\n", 
    "w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]\n", 
    "w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]\n", 
    "w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]\n", 
    "\n", 
    "# Second filter detects horizontal edges in the blue channel.\n", 
    "w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]\n", 
    "\n", 
    "# Vector of biases. We don't need any bias for the grayscale\n", 
    "# filter, but for the edge detection filter we want to add 128\n", 
    "# to each output so that nothing is negative.\n", 
    "b = np.array([0, 128])\n", 
    "\n", 
    "# Compute the result of convolving each input in x with each filter in w,\n", 
    "# offsetting by b, and storing the results in out.\n", 
    "out, _ = conv_forward_naive(x, w, b, {'stride': 1, 'pad': 1})\n", 
    "\n", 
    "def imshow_noax(img, normalize=True):\n", 
    "    \"\"\" Tiny helper to show images as uint8 and remove axis labels \"\"\"\n", 
    "    if normalize:\n", 
    "        img_max, img_min = np.max(img), np.min(img)\n", 
    "        img = 255.0 * (img - img_min) / (img_max - img_min)\n", 
    "    plt.imshow(img.astype('uint8'))\n", 
    "    plt.gca().axis('off')\n", 
    "\n", 
    "# Show the original images and the results of the conv operation\n", 
    "plt.subplot(2, 3, 1)\n", 
    "imshow_noax(puppy, normalize=False)\n", 
    "plt.title('Original image')\n", 
    "plt.subplot(2, 3, 2)\n", 
    "imshow_noax(out[0, 0])\n", 
    "plt.title('Grayscale')\n", 
    "plt.subplot(2, 3, 3)\n", 
    "imshow_noax(out[0, 1])\n", 
    "plt.title('Edges')\n", 
    "plt.subplot(2, 3, 4)\n", 
    "imshow_noax(kitten_cropped, normalize=False)\n", 
    "plt.subplot(2, 3, 5)\n", 
    "imshow_noax(out[1, 0])\n", 
    "plt.subplot(2, 3, 6)\n", 
    "imshow_noax(out[1, 1])\n", 
    "plt.show()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Convolution: Naive backward pass\n", 
    "Implement the backward pass for the convolution operation in the function `conv_backward_naive` in the file `cs231n/layers.py`. Again, you don't need to worry too much about computational efficiency.\n", 
    "\n", 
    "When you are done, run the following to check your backward pass with a numeric gradient check."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "x = np.random.randn(4, 3, 5, 5)\n", 
    "w = np.random.randn(2, 3, 3, 3)\n", 
    "b = np.random.randn(2,)\n", 
    "dout = np.random.randn(4, 2, 5, 5)\n", 
    "conv_param = {'stride': 1, 'pad': 1}\n", 
    "\n", 
    "dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x, w, b, conv_param)[0], x, dout)\n", 
    "dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x, w, b, conv_param)[0], w, dout)\n", 
    "db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x, w, b, conv_param)[0], b, dout)\n", 
    "\n", 
    "out, cache = conv_forward_naive(x, w, b, conv_param)\n", 
    "dx, dw, db = conv_backward_naive(dout, cache)\n", 
    "\n", 
    "# Your errors should be around 1e-9'\n", 
    "print 'Testing conv_backward_naive function'\n", 
    "print 'dx error: ', rel_error(dx, dx_num)\n", 
    "print 'dw error: ', rel_error(dw, dw_num)\n", 
    "print 'db error: ', rel_error(db, db_num)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Max pooling: Naive forward\n", 
    "Implement the forward pass for the max-pooling operation in the function `max_pool_forward_naive` in the file `cs231n/layers.py`. Again, don't worry too much about computational efficiency.\n", 
    "\n", 
    "Check your implementation by running the following:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "x_shape = (2, 3, 4, 4)\n", 
    "x = np.linspace(-0.3, 0.4, num=np.prod(x_shape)).reshape(x_shape)\n", 
    "pool_param = {'pool_width': 2, 'pool_height': 2, 'stride': 2}\n", 
    "\n", 
    "out, _ = max_pool_forward_naive(x, pool_param)\n", 
    "\n", 
    "correct_out = np.array([[[[-0.26315789, -0.24842105],\n", 
    "                          [-0.20421053, -0.18947368]],\n", 
    "                         [[-0.14526316, -0.13052632],\n", 
    "                          [-0.08631579, -0.07157895]],\n", 
    "                         [[-0.02736842, -0.01263158],\n", 
    "                          [ 0.03157895,  0.04631579]]],\n", 
    "                        [[[ 0.09052632,  0.10526316],\n", 
    "                          [ 0.14947368,  0.16421053]],\n", 
    "                         [[ 0.20842105,  0.22315789],\n", 
    "                          [ 0.26736842,  0.28210526]],\n", 
    "                         [[ 0.32631579,  0.34105263],\n", 
    "                          [ 0.38526316,  0.4       ]]]])\n", 
    "\n", 
    "# Compare your output with ours. Difference should be around 1e-8.\n", 
    "print 'Testing max_pool_forward_naive function:'\n", 
    "print 'difference: ', rel_error(out, correct_out)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Max pooling: Naive backward\n", 
    "Implement the backward pass for the max-pooling operation in the function `max_pool_backward_naive` in the file `cs231n/layers.py`. You don't need to worry about computational efficiency.\n", 
    "\n", 
    "Check your implementation with numeric gradient checking by running the following:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "x = np.random.randn(3, 2, 8, 8)\n", 
    "dout = np.random.randn(3, 2, 4, 4)\n", 
    "pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}\n", 
    "\n", 
    "dx_num = eval_numerical_gradient_array(lambda x: max_pool_forward_naive(x, pool_param)[0], x, dout)\n", 
    "\n", 
    "out, cache = max_pool_forward_naive(x, pool_param)\n", 
    "dx = max_pool_backward_naive(dout, cache)\n", 
    "\n", 
    "# Your error should be around 1e-12\n", 
    "print 'Testing max_pool_backward_naive function:'\n", 
    "print 'dx error: ', rel_error(dx, dx_num)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Fast layers\n", 
    "Making convolution and pooling layers fast can be challenging. To spare you the pain, we've provided fast implementations of the forward and backward passes for convolution and pooling layers in the file `cs231n/fast_layers.py`.\n", 
    "\n", 
    "The fast convolution implementation depends on a Cython extension; to compile it you need to run the following from the `cs231n` directory:\n", 
    "\n", 
    "```bash\n", 
    "python setup.py build_ext --inplace\n", 
    "```\n", 
    "\n", 
    "The API for the fast versions of the convolution and pooling layers is exactly the same as the naive versions that you implemented above: the forward pass receives data, weights, and parameters and produces outputs and a cache object; the backward pass recieves upstream derivatives and the cache object and produces gradients with respect to the data and weights.\n", 
    "\n", 
    "**NOTE:** The fast implementation for pooling will only perform optimally if the pooling regions are non-overlapping and tile the input. If these conditions are not met then the fast pooling implementation will not be much faster than the naive implementation.\n", 
    "\n", 
    "You can compare the performance of the naive and fast versions of these layers by running the following:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "from cs231n.fast_layers import conv_forward_fast, conv_backward_fast\n", 
    "from time import time\n", 
    "\n", 
    "x = np.random.randn(100, 3, 31, 31)\n", 
    "w = np.random.randn(25, 3, 3, 3)\n", 
    "b = np.random.randn(25,)\n", 
    "dout = np.random.randn(100, 25, 16, 16)\n", 
    "conv_param = {'stride': 2, 'pad': 1}\n", 
    "\n", 
    "t0 = time()\n", 
    "out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param)\n", 
    "t1 = time()\n", 
    "out_fast, cache_fast = conv_forward_fast(x, w, b, conv_param)\n", 
    "t2 = time()\n", 
    "\n", 
    "print 'Testing conv_forward_fast:'\n", 
    "print 'Naive: %fs' % (t1 - t0)\n", 
    "print 'Fast: %fs' % (t2 - t1)\n", 
    "print 'Speedup: %fx' % ((t1 - t0) / (t2 - t1))\n", 
    "print 'Difference: ', rel_error(out_naive, out_fast)\n", 
    "\n", 
    "t0 = time()\n", 
    "dx_naive, dw_naive, db_naive = conv_backward_naive(dout, cache_naive)\n", 
    "t1 = time()\n", 
    "dx_fast, dw_fast, db_fast = conv_backward_fast(dout, cache_fast)\n", 
    "t2 = time()\n", 
    "\n", 
    "print '\\nTesting conv_backward_fast:'\n", 
    "print 'Naive: %fs' % (t1 - t0)\n", 
    "print 'Fast: %fs' % (t2 - t1)\n", 
    "print 'Speedup: %fx' % ((t1 - t0) / (t2 - t1))\n", 
    "print 'dx difference: ', rel_error(dx_naive, dx_fast)\n", 
    "print 'dw difference: ', rel_error(dw_naive, dw_fast)\n", 
    "print 'db difference: ', rel_error(db_naive, db_fast)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "from cs231n.fast_layers import max_pool_forward_fast, max_pool_backward_fast\n", 
    "\n", 
    "x = np.random.randn(100, 3, 32, 32)\n", 
    "dout = np.random.randn(100, 3, 16, 16)\n", 
    "pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}\n", 
    "\n", 
    "t0 = time()\n", 
    "out_naive, cache_naive = max_pool_forward_naive(x, pool_param)\n", 
    "t1 = time()\n", 
    "out_fast, cache_fast = max_pool_forward_fast(x, pool_param)\n", 
    "t2 = time()\n", 
    "\n", 
    "print 'Testing pool_forward_fast:'\n", 
    "print 'Naive: %fs' % (t1 - t0)\n", 
    "print 'fast: %fs' % (t2 - t1)\n", 
    "print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))\n", 
    "print 'difference: ', rel_error(out_naive, out_fast)\n", 
    "\n", 
    "t0 = time()\n", 
    "dx_naive = max_pool_backward_naive(dout, cache_naive)\n", 
    "t1 = time()\n", 
    "dx_fast = max_pool_backward_fast(dout, cache_fast)\n", 
    "t2 = time()\n", 
    "\n", 
    "print '\\nTesting pool_backward_fast:'\n", 
    "print 'Naive: %fs' % (t1 - t0)\n", 
    "print 'speedup: %fx' % ((t1 - t0) / (t2 - t1))\n", 
    "print 'dx difference: ', rel_error(dx_naive, dx_fast)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Convolutional \"sandwich\" layers\n", 
    "Previously we introduced the concept of \"sandwich\" layers that combine multiple operations into commonly used patterns. In the file `cs231n/layer_utils.py` you will find sandwich layers that implement a few commonly used patterns for convolutional networks."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "from cs231n.layer_utils import conv_relu_pool_forward, conv_relu_pool_backward\n", 
    "\n", 
    "x = np.random.randn(2, 3, 16, 16)\n", 
    "w = np.random.randn(3, 3, 3, 3)\n", 
    "b = np.random.randn(3,)\n", 
    "dout = np.random.randn(2, 3, 8, 8)\n", 
    "conv_param = {'stride': 1, 'pad': 1}\n", 
    "pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}\n", 
    "\n", 
    "out, cache = conv_relu_pool_forward(x, w, b, conv_param, pool_param)\n", 
    "dx, dw, db = conv_relu_pool_backward(dout, cache)\n", 
    "\n", 
    "dx_num = eval_numerical_gradient_array(lambda x: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], x, dout)\n", 
    "dw_num = eval_numerical_gradient_array(lambda w: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], w, dout)\n", 
    "db_num = eval_numerical_gradient_array(lambda b: conv_relu_pool_forward(x, w, b, conv_param, pool_param)[0], b, dout)\n", 
    "\n", 
    "print 'Testing conv_relu_pool'\n", 
    "print 'dx error: ', rel_error(dx_num, dx)\n", 
    "print 'dw error: ', rel_error(dw_num, dw)\n", 
    "print 'db error: ', rel_error(db_num, db)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "from cs231n.layer_utils import conv_relu_forward, conv_relu_backward\n", 
    "\n", 
    "x = np.random.randn(2, 3, 8, 8)\n", 
    "w = np.random.randn(3, 3, 3, 3)\n", 
    "b = np.random.randn(3,)\n", 
    "dout = np.random.randn(2, 3, 8, 8)\n", 
    "conv_param = {'stride': 1, 'pad': 1}\n", 
    "\n", 
    "out, cache = conv_relu_forward(x, w, b, conv_param)\n", 
    "dx, dw, db = conv_relu_backward(dout, cache)\n", 
    "\n", 
    "dx_num = eval_numerical_gradient_array(lambda x: conv_relu_forward(x, w, b, conv_param)[0], x, dout)\n", 
    "dw_num = eval_numerical_gradient_array(lambda w: conv_relu_forward(x, w, b, conv_param)[0], w, dout)\n", 
    "db_num = eval_numerical_gradient_array(lambda b: conv_relu_forward(x, w, b, conv_param)[0], b, dout)\n", 
    "\n", 
    "print 'Testing conv_relu:'\n", 
    "print 'dx error: ', rel_error(dx_num, dx)\n", 
    "print 'dw error: ', rel_error(dw_num, dw)\n", 
    "print 'db error: ', rel_error(db_num, db)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Three-layer ConvNet\n", 
    "Now that you have implemented all the necessary layers, we can put them together into a simple convolutional network.\n", 
    "\n", 
    "Open the file `cs231n/cnn.py` and complete the implementation of the `ThreeLayerConvNet` class. Run the following cells to help you debug:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "source": [
    "## Sanity check loss\n", 
    "After you build a new network, one of the first things you should do is sanity check the loss. When we use the softmax loss, we expect the loss for random weights (and no regularization) to be about `log(C)` for `C` classes. When we add regularization this should go up."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "model = ThreeLayerConvNet()\n", 
    "\n", 
    "N = 50\n", 
    "X = np.random.randn(N, 3, 32, 32)\n", 
    "y = np.random.randint(10, size=N)\n", 
    "\n", 
    "loss, grads = model.loss(X, y)\n", 
    "print 'Initial loss (no regularization): ', loss\n", 
    "\n", 
    "model.reg = 0.5\n", 
    "loss, grads = model.loss(X, y)\n", 
    "print 'Initial loss (with regularization): ', loss"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "## Gradient check\n", 
    "After the loss looks reasonable, use numeric gradient checking to make sure that your backward pass is correct. When you use numeric gradient checking you should use a small amount of artifical data and a small number of neurons at each layer."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "num_inputs = 2\n", 
    "input_dim = (3, 16, 16)\n", 
    "reg = 0.0\n", 
    "num_classes = 10\n", 
    "X = np.random.randn(num_inputs, *input_dim)\n", 
    "y = np.random.randint(num_classes, size=num_inputs)\n", 
    "\n", 
    "model = ThreeLayerConvNet(num_filters=3, filter_size=3,\n", 
    "                          input_dim=input_dim, hidden_dim=7,\n", 
    "                          dtype=np.float64)\n", 
    "loss, grads = model.loss(X, y)\n", 
    "for param_name in sorted(grads):\n", 
    "    f = lambda _: model.loss(X, y)[0]\n", 
    "    param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)\n", 
    "    e = rel_error(param_grad_num, grads[param_name])\n", 
    "    print '%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name]))"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "## Overfit small data\n", 
    "A nice trick is to train your model with just a few training samples. You should be able to overfit small datasets, which will result in very high training accuracy and comparatively low validation accuracy."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "num_train = 100\n", 
    "small_data = {\n", 
    "  'X_train': data['X_train'][:num_train],\n", 
    "  'y_train': data['y_train'][:num_train],\n", 
    "  'X_val': data['X_val'],\n", 
    "  'y_val': data['y_val'],\n", 
    "}\n", 
    "\n", 
    "model = ThreeLayerConvNet(weight_scale=1e-2)\n", 
    "\n", 
    "solver = Solver(model, small_data,\n", 
    "                num_epochs=10, batch_size=50,\n", 
    "                update_rule='adam',\n", 
    "                optim_config={\n", 
    "                  'learning_rate': 1e-3,\n", 
    "                },\n", 
    "                verbose=True, print_every=1)\n", 
    "solver.train()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "Plotting the loss, training accuracy, and validation accuracy should show clear overfitting:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "plt.subplot(2, 1, 1)\n", 
    "plt.plot(solver.loss_history, 'o')\n", 
    "plt.xlabel('iteration')\n", 
    "plt.ylabel('loss')\n", 
    "\n", 
    "plt.subplot(2, 1, 2)\n", 
    "plt.plot(solver.train_acc_history, '-o')\n", 
    "plt.plot(solver.val_acc_history, '-o')\n", 
    "plt.legend(['train', 'val'], loc='upper left')\n", 
    "plt.xlabel('epoch')\n", 
    "plt.ylabel('accuracy')\n", 
    "plt.show()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "## Train the net\n", 
    "By training the three-layer convolutional network for one epoch, you should achieve greater than 40% accuracy on the training set:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "model = ThreeLayerConvNet(weight_scale=0.001, hidden_dim=500, reg=0.001)\n", 
    "\n", 
    "solver = Solver(model, data,\n", 
    "                num_epochs=1, batch_size=50,\n", 
    "                update_rule='adam',\n", 
    "                optim_config={\n", 
    "                  'learning_rate': 1e-3,\n", 
    "                },\n", 
    "                verbose=True, print_every=20)\n", 
    "solver.train()"
   ], 
   "outputs": [], 
   "metadata": {
    "scrolled": false, 
    "collapsed": false
   }
  }, 
  {
   "source": [
    "## Visualize Filters\n", 
    "You can visualize the first-layer convolutional filters from the trained network by running the following:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "from cs231n.vis_utils import visualize_grid\n", 
    "\n", 
    "grid = visualize_grid(model.params['W1'].transpose(0, 2, 3, 1))\n", 
    "plt.imshow(grid.astype('uint8'))\n", 
    "plt.axis('off')\n", 
    "plt.gcf().set_size_inches(5, 5)\n", 
    "plt.show()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Spatial Batch Normalization\n", 
    "We already saw that batch normalization is a very useful technique for training deep fully-connected networks. Batch normalization can also be used for convolutional networks, but we need to tweak it a bit; the modification will be called \"spatial batch normalization.\"\n", 
    "\n", 
    "Normally batch-normalization accepts inputs of shape `(N, D)` and produces outputs of shape `(N, D)`, where we normalize across the minibatch dimension `N`. For data coming from convolutional layers, batch normalization needs to accept inputs of shape `(N, C, H, W)` and produce outputs of shape `(N, C, H, W)` where the `N` dimension gives the minibatch size and the `(H, W)` dimensions give the spatial size of the feature map.\n", 
    "\n", 
    "If the feature map was produced using convolutions, then we expect the statistics of each feature channel to be relatively consistent both between different imagesand different locations within the same image. Therefore spatial batch normalization computes a mean and variance for each of the `C` feature channels by computing statistics over both the minibatch dimension `N` and the spatial dimensions `H` and `W`."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "source": [
    "## Spatial batch normalization: forward\n", 
    "\n", 
    "In the file `cs231n/layers.py`, implement the forward pass for spatial batch normalization in the function `spatial_batchnorm_forward`. Check your implementation by running the following:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Check the training-time forward pass by checking means and variances\n", 
    "# of features both before and after spatial batch normalization\n", 
    "\n", 
    "N, C, H, W = 2, 3, 4, 5\n", 
    "x = 4 * np.random.randn(N, C, H, W) + 10\n", 
    "\n", 
    "print 'Before spatial batch normalization:'\n", 
    "print '  Shape: ', x.shape\n", 
    "print '  Means: ', x.mean(axis=(0, 2, 3))\n", 
    "print '  Stds: ', x.std(axis=(0, 2, 3))\n", 
    "\n", 
    "# Means should be close to zero and stds close to one\n", 
    "gamma, beta = np.ones(C), np.zeros(C)\n", 
    "bn_param = {'mode': 'train'}\n", 
    "out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)\n", 
    "print 'After spatial batch normalization:'\n", 
    "print '  Shape: ', out.shape\n", 
    "print '  Means: ', out.mean(axis=(0, 2, 3))\n", 
    "print '  Stds: ', out.std(axis=(0, 2, 3))\n", 
    "\n", 
    "# Means should be close to beta and stds close to gamma\n", 
    "gamma, beta = np.asarray([3, 4, 5]), np.asarray([6, 7, 8])\n", 
    "out, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)\n", 
    "print 'After spatial batch normalization (nontrivial gamma, beta):'\n", 
    "print '  Shape: ', out.shape\n", 
    "print '  Means: ', out.mean(axis=(0, 2, 3))\n", 
    "print '  Stds: ', out.std(axis=(0, 2, 3))"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Check the test-time forward pass by running the training-time\n", 
    "# forward pass many times to warm up the running averages, and then\n", 
    "# checking the means and variances of activations after a test-time\n", 
    "# forward pass.\n", 
    "\n", 
    "N, C, H, W = 10, 4, 11, 12\n", 
    "\n", 
    "bn_param = {'mode': 'train'}\n", 
    "gamma = np.ones(C)\n", 
    "beta = np.zeros(C)\n", 
    "for t in xrange(50):\n", 
    "  x = 2.3 * np.random.randn(N, C, H, W) + 13\n", 
    "  spatial_batchnorm_forward(x, gamma, beta, bn_param)\n", 
    "bn_param['mode'] = 'test'\n", 
    "x = 2.3 * np.random.randn(N, C, H, W) + 13\n", 
    "a_norm, _ = spatial_batchnorm_forward(x, gamma, beta, bn_param)\n", 
    "\n", 
    "# Means should be close to zero and stds close to one, but will be\n", 
    "# noisier than training-time forward passes.\n", 
    "print 'After spatial batch normalization (test-time):'\n", 
    "print '  means: ', a_norm.mean(axis=(0, 2, 3))\n", 
    "print '  stds: ', a_norm.std(axis=(0, 2, 3))"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "## Spatial batch normalization: backward\n", 
    "In the file `cs231n/layers.py`, implement the backward pass for spatial batch normalization in the function `spatial_batchnorm_backward`. Run the following to check your implementation using a numeric gradient check:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, C, H, W = 2, 3, 4, 5\n", 
    "x = 5 * np.random.randn(N, C, H, W) + 12\n", 
    "gamma = np.random.randn(C)\n", 
    "beta = np.random.randn(C)\n", 
    "dout = np.random.randn(N, C, H, W)\n", 
    "\n", 
    "bn_param = {'mode': 'train'}\n", 
    "fx = lambda x: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]\n", 
    "fg = lambda a: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]\n", 
    "fb = lambda b: spatial_batchnorm_forward(x, gamma, beta, bn_param)[0]\n", 
    "\n", 
    "dx_num = eval_numerical_gradient_array(fx, x, dout)\n", 
    "da_num = eval_numerical_gradient_array(fg, gamma, dout)\n", 
    "db_num = eval_numerical_gradient_array(fb, beta, dout)\n", 
    "\n", 
    "_, cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)\n", 
    "dx, dgamma, dbeta = spatial_batchnorm_backward(dout, cache)\n", 
    "print 'dx error: ', rel_error(dx_num, dx)\n", 
    "print 'dgamma error: ', rel_error(da_num, dgamma)\n", 
    "print 'dbeta error: ', rel_error(db_num, dbeta)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Experiment!\n", 
    "Experiment and try to get the best performance that you can on CIFAR-10 using a ConvNet. Here are some ideas to get you started:\n", 
    "\n", 
    "### Things you should try:\n", 
    "- Filter size: Above we used 7x7; this makes pretty pictures but smaller filters may be more efficient\n", 
    "- Number of filters: Above we used 32 filters. Do more or fewer do better?\n", 
    "- Batch normalization: Try adding spatial batch normalization after convolution layers and vanilla batch normalization aafter affine layers. Do your networks train faster?\n", 
    "- Network architecture: The network above has two layers of trainable parameters. Can you do better with a deeper network? You can implement alternative architectures in the file `cs231n/classifiers/convnet.py`. Some good architectures to try include:\n", 
    "    - [conv-relu-pool]xN - conv - relu - [affine]xM - [softmax or SVM]\n", 
    "    - [conv-relu-pool]XN - [affine]XM - [softmax or SVM]\n", 
    "    - [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]\n", 
    "\n", 
    "### Tips for training\n", 
    "For each network architecture that you try, you should tune the learning rate and regularization strength. When doing this there are a couple important things to keep in mind:\n", 
    "\n", 
    "- If the parameters are working well, you should see improvement within a few hundred iterations\n", 
    "- Remember the course-to-fine approach for hyperparameter tuning: start by testing a large range of hyperparameters for just a few training iterations to find the combinations of parameters that are working at all.\n", 
    "- Once you have found some sets of parameters that seem to work, search more finely around these parameters. You may need to train for more epochs.\n", 
    "\n", 
    "### Going above and beyond\n", 
    "If you are feeling adventurous there are many other features you can implement to try and improve your performance. You are **not required** to implement any of these; however they would be good things to try for extra credit.\n", 
    "\n", 
    "- Alternative update steps: For the assignment we implemented SGD+momentum, RMSprop, and Adam; you could try alternatives like AdaGrad or AdaDelta.\n", 
    "- Alternative activation functions such as leaky ReLU, parametric ReLU, or MaxOut.\n", 
    "- Model ensembles\n", 
    "- Data augmentation\n", 
    "\n", 
    "If you do decide to implement something extra, clearly describe it in the \"Extra Credit Description\" cell below.\n", 
    "\n", 
    "### What we expect\n", 
    "At the very least, you should be able to train a ConvNet that gets at least 65% accuracy on the validation set. This is just a lower bound - if you are careful it should be possible to get accuracies much higher than that! Extra credit points will be awarded for particularly high-scoring models or unique approaches.\n", 
    "\n", 
    "You should use the space below to experiment and train your network. The final cell in this notebook should contain the training, validation, and test set accuracies for your final trained network. In this notebook you should also write an explanation of what you did, any additional features that you implemented, and any visualizations or graphs that you make in the process of training and evaluating your network.\n", 
    "\n", 
    "Have fun and happy training!"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Train a really good model on CIFAR-10"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": true
   }
  }, 
  {
   "source": [
    "# Extra Credit Description\n", 
    "If you implement any additional features for extra credit, clearly describe them here with pointers to any code in this or other files if applicable."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
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