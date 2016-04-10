{
 "nbformat_minor": 0, 
 "nbformat": 4, 
 "cells": [
  {
   "source": [
    "# Image Generation\n", 
    "In this notebook we will continue our exploration of image gradients using the deep model that was pretrained on TinyImageNet. We will explore various ways of using these image gradients to generate images. We will implement class visualizations, feature inversion, and DeepDream."
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
    "from scipy.misc import imread, imresize\n", 
    "import matplotlib.pyplot as plt\n", 
    "\n", 
    "from cs231n.classifiers.pretrained_cnn import PretrainedCNN\n", 
    "from cs231n.data_utils import load_tiny_imagenet\n", 
    "from cs231n.image_utils import blur_image, deprocess_image, preprocess_image\n", 
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
    "# TinyImageNet and pretrained model\n", 
    "As in the previous notebook, load the TinyImageNet dataset and the pretrained model."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "data = load_tiny_imagenet('cs231n/datasets/tiny-imagenet-100-A', subtract_mean=True)\n", 
    "model = PretrainedCNN(h5_file='cs231n/datasets/pretrained_model.h5')"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    " # Class visualization\n", 
    "By starting with a random noise image and performing gradient ascent on a target class, we can generate an image that the network will recognize as the target class. This idea was first presented in [1]; [2] extended this idea by suggesting several regularization techniques that can improve the quality of the generated image.\n", 
    "\n", 
    "Concretely, let $I$ be an image and let $y$ be a target class. Let $s_y(I)$ be the score that a convolutional network assigns to the image $I$ for class $y$; note that these are raw unnormalized scores, not class probabilities. We wish to generate an image $I^*$ that achieves a high score for the class $y$ by solving the problem\n", 
    "\n", 
    "$$\n", 
    "I^* = \\arg\\max_I s_y(I) + R(I)\n", 
    "$$\n", 
    "\n", 
    "where $R$ is a (possibly implicit) regularizer. We can solve this optimization problem using gradient descent, computing gradients with respect to the generated image. We will use (explicit) L2 regularization of the form\n", 
    "\n", 
    "$$\n", 
    "R(I) + \\lambda \\|I\\|_2^2\n", 
    "$$\n", 
    "\n", 
    "and implicit regularization as suggested by [2] by peridically blurring the generated image. We can solve this problem using gradient ascent on the generated image.\n", 
    "\n", 
    "In the cell below, complete the implementation of the `create_class_visualization` function.\n", 
    "\n", 
    "[1] Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. \"Deep Inside Convolutional Networks: Visualising\n", 
    "Image Classification Models and Saliency Maps\", ICLR Workshop 2014.\n", 
    "\n", 
    "[2] Yosinski et al, \"Understanding Neural Networks Through Deep Visualization\", ICML 2015 Deep Learning Workshop"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "def create_class_visualization(target_y, model, **kwargs):\n", 
    "  \"\"\"\n", 
    "  Perform optimization over the image to generate class visualizations.\n", 
    "  \n", 
    "  Inputs:\n", 
    "  - target_y: Integer in the range [0, 100) giving the target class\n", 
    "  - model: A PretrainedCNN that will be used for generation\n", 
    "  \n", 
    "  Keyword arguments:\n", 
    "  - learning_rate: Floating point number giving the learning rate\n", 
    "  - blur_every: An integer; how often to blur the image as a regularizer\n", 
    "  - l2_reg: Floating point number giving L2 regularization strength on the image;\n", 
    "    this is lambda in the equation above.\n", 
    "  - max_jitter: How much random jitter to add to the image as regularization\n", 
    "  - num_iterations: How many iterations to run for\n", 
    "  - show_every: How often to show the image\n", 
    "  \"\"\"\n", 
    "  \n", 
    "  learning_rate = kwargs.pop('learning_rate', 10000)\n", 
    "  blur_every = kwargs.pop('blur_every', 1)\n", 
    "  l2_reg = kwargs.pop('l2_reg', 1e-6)\n", 
    "  max_jitter = kwargs.pop('max_jitter', 4)\n", 
    "  num_iterations = kwargs.pop('num_iterations', 100)\n", 
    "  show_every = kwargs.pop('show_every', 25)\n", 
    "  \n", 
    "  X = np.random.randn(1, 3, 64, 64)\n", 
    "  for t in xrange(num_iterations):\n", 
    "    # As a regularizer, add random jitter to the image\n", 
    "    ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)\n", 
    "    X = np.roll(np.roll(X, ox, -1), oy, -2)\n", 
    "\n", 
    "    dX = None\n", 
    "    ############################################################################\n", 
    "    # TODO: Compute the image gradient dX of the image with respect to the     #\n", 
    "    # target_y class score. This should be similar to the fooling images. Also #\n", 
    "    # add L2 regularization to dX and update the image X using the image       #\n", 
    "    # gradient and the learning rate.                                          #\n", 
    "    ############################################################################\n", 
    "    pass\n", 
    "    ############################################################################\n", 
    "    #                             END OF YOUR CODE                             #\n", 
    "    ############################################################################\n", 
    "    \n", 
    "    # Undo the jitter\n", 
    "    X = np.roll(np.roll(X, -ox, -1), -oy, -2)\n", 
    "    \n", 
    "    # As a regularizer, clip the image\n", 
    "    X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])\n", 
    "    \n", 
    "    # As a regularizer, periodically blur the image\n", 
    "    if t % blur_every == 0:\n", 
    "      X = blur_image(X)\n", 
    "    \n", 
    "    # Periodically show the image\n", 
    "    if t % show_every == 0:\n", 
    "      plt.imshow(deprocess_image(X, data['mean_image']))\n", 
    "      plt.gcf().set_size_inches(3, 3)\n", 
    "      plt.axis('off')\n", 
    "      plt.show()\n", 
    "  return X"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": true
   }
  }, 
  {
   "source": [
    "You can use the code above to generate some cool images! An example is shown below. Try to generate a cool-looking image. If you want you can try to implement the other regularization schemes from Yosinski et al, but it isn't required."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "target_y = 43 # Tarantula\n", 
    "print data['class_names'][target_y]\n", 
    "X = create_class_visualization(target_y, model, show_every=25)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Feature Inversion\n", 
    "In an attempt to understand the types of features that convolutional networks learn to recognize, a recent paper [1] attempts to reconstruct an image from its feature representation. We can easily implement this idea using image gradients from the pretrained network.\n", 
    "\n", 
    "Concretely, given a image $I$, let $\\phi_\\ell(I)$ be the activations at layer $\\ell$ of the convolutional network $\\phi$. We wish to find an image $I^*$ with a similar feature representation as $I$ at layer $\\ell$ of the network $\\phi$ by solving the optimization problem\n", 
    "\n", 
    "$$\n", 
    "I^* = \\arg\\min_{I'} \\|\\phi_\\ell(I) - \\phi_\\ell(I')\\|_2^2 + R(I')\n", 
    "$$\n", 
    "\n", 
    "where $\\|\\cdot\\|_2^2$ is the squared Euclidean norm. As above, $R$ is a (possibly implicit) regularizer. We can solve this optimization problem using gradient descent, computing gradients with respect to the generated image. We will use (explicit) L2 regularization of the form\n", 
    "\n", 
    "$$\n", 
    "R(I') + \\lambda \\|I'\\|_2^2\n", 
    "$$\n", 
    "\n", 
    "together with implicit regularization by periodically blurring the image, as recommended by [2].\n", 
    "\n", 
    "Implement this method in the function below.\n", 
    "\n", 
    "[1] Aravindh Mahendran, Andrea Vedaldi, \"Understanding Deep Image Representations by Inverting them\", CVPR 2015\n", 
    "\n", 
    "[2] Yosinski et al, \"Understanding Neural Networks Through Deep Visualization\", ICML 2015 Deep Learning Workshop"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "def invert_features(target_feats, layer, model, **kwargs):\n", 
    "  \"\"\"\n", 
    "  Perform feature inversion in the style of Mahendran and Vedaldi 2015, using\n", 
    "  L2 regularization and periodic blurring.\n", 
    "  \n", 
    "  Inputs:\n", 
    "  - target_feats: Image features of the target image, of shape (1, C, H, W);\n", 
    "    we will try to generate an image that matches these features\n", 
    "  - layer: The index of the layer from which the features were extracted\n", 
    "  - model: A PretrainedCNN that was used to extract features\n", 
    "  \n", 
    "  Keyword arguments:\n", 
    "  - learning_rate: The learning rate to use for gradient descent\n", 
    "  - num_iterations: The number of iterations to use for gradient descent\n", 
    "  - l2_reg: The strength of L2 regularization to use; this is lambda in the\n", 
    "    equation above.\n", 
    "  - blur_every: How often to blur the image as implicit regularization; set\n", 
    "    to 0 to disable blurring.\n", 
    "  - show_every: How often to show the generated image; set to 0 to disable\n", 
    "    showing intermediate reuslts.\n", 
    "    \n", 
    "  Returns:\n", 
    "  - X: Generated image of shape (1, 3, 64, 64) that matches the target features.\n", 
    "  \"\"\"\n", 
    "  learning_rate = kwargs.pop('learning_rate', 10000)\n", 
    "  num_iterations = kwargs.pop('num_iterations', 500)\n", 
    "  l2_reg = kwargs.pop('l2_reg', 1e-7)\n", 
    "  blur_every = kwargs.pop('blur_every', 1)\n", 
    "  show_every = kwargs.pop('show_every', 50)\n", 
    "  \n", 
    "  X = np.random.randn(1, 3, 64, 64)\n", 
    "  for t in xrange(num_iterations):\n", 
    "    ############################################################################\n", 
    "    # TODO: Compute the image gradient dX of the reconstruction loss with      #\n", 
    "    # respect to the image. You should include L2 regularization penalizing    #\n", 
    "    # large pixel values in the generated image using the l2_reg parameter;    #\n", 
    "    # then update the generated image using the learning_rate from above.      #\n", 
    "    ############################################################################\n", 
    "    pass\n", 
    "    ############################################################################\n", 
    "    #                             END OF YOUR CODE                             #\n", 
    "    ############################################################################\n", 
    "    \n", 
    "    # As a regularizer, clip the image\n", 
    "    X = np.clip(X, -data['mean_image'], 255.0 - data['mean_image'])\n", 
    "    \n", 
    "    # As a regularizer, periodically blur the image\n", 
    "    if (blur_every > 0) and t % blur_every == 0:\n", 
    "      X = blur_image(X)\n", 
    "\n", 
    "    if (show_every > 0) and (t % show_every == 0 or t + 1 == num_iterations):\n", 
    "      plt.imshow(deprocess_image(X, data['mean_image']))\n", 
    "      plt.gcf().set_size_inches(3, 3)\n", 
    "      plt.axis('off')\n", 
    "      plt.title('t = %d' % t)\n", 
    "      plt.show()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "### Shallow feature reconstruction\n", 
    "After implementing the feature inversion above, run the following cell to try and reconstruct features from the fourth convolutional layer of the pretrained model. You should be able to reconstruct the features using the provided optimization parameters."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "filename = 'kitten.jpg'\n", 
    "layer = 3 # layers start from 0 so these are features after 4 convolutions\n", 
    "img = imresize(imread(filename), (64, 64))\n", 
    "\n", 
    "plt.imshow(img)\n", 
    "plt.gcf().set_size_inches(3, 3)\n", 
    "plt.title('Original image')\n", 
    "plt.axis('off')\n", 
    "plt.show()\n", 
    "\n", 
    "# Preprocess the image before passing it to the network:\n", 
    "# subtract the mean, add a dimension, etc\n", 
    "img_pre = preprocess_image(img, data['mean_image'])\n", 
    "\n", 
    "# Extract features from the image\n", 
    "feats, _ = model.forward(img_pre, end=layer)\n", 
    "\n", 
    "# Invert the features\n", 
    "kwargs = {\n", 
    "  'num_iterations': 400,\n", 
    "  'learning_rate': 5000,\n", 
    "  'l2_reg': 1e-8,\n", 
    "  'show_every': 100,\n", 
    "  'blur_every': 10,\n", 
    "}\n", 
    "X = invert_features(feats, layer, model, **kwargs)"
   ], 
   "outputs": [], 
   "metadata": {
    "scrolled": false, 
    "collapsed": false
   }
  }, 
  {
   "source": [
    "### Deep feature reconstruction\n", 
    "Reconstructing images using features from deeper layers of the network tends to give interesting results. In the cell below, try to reconstruct the best image you can by inverting the features after 7 layers of convolutions. You will need to play with the hyperparameters to try and get a good result.\n", 
    "\n", 
    "HINT: If you read the paper by Mahendran and Vedaldi, you'll see that reconstructions from deep features tend not to look much like the original image, so you shouldn't expect the results to look like the reconstruction above. You should be able to get an image that shows some discernable structure within 1000 iterations."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "filename = 'kitten.jpg'\n", 
    "layer = 6 # layers start from 0 so these are features after 7 convolutions\n", 
    "img = imresize(imread(filename), (64, 64))\n", 
    "\n", 
    "plt.imshow(img)\n", 
    "plt.gcf().set_size_inches(3, 3)\n", 
    "plt.title('Original image')\n", 
    "plt.axis('off')\n", 
    "plt.show()\n", 
    "\n", 
    "# Preprocess the image before passing it to the network:\n", 
    "# subtract the mean, add a dimension, etc\n", 
    "img_pre = preprocess_image(img, data['mean_image'])\n", 
    "\n", 
    "# Extract features from the image\n", 
    "feats, _ = model.forward(img_pre, end=layer)\n", 
    "\n", 
    "# Invert the features\n", 
    "# You will need to play with these parameters.\n", 
    "kwargs = {\n", 
    "  'num_iterations': 1000,\n", 
    "  'learning_rate': 0,\n", 
    "  'l2_reg': 0,\n", 
    "  'show_every': 100,\n", 
    "  'blur_every': 0,\n", 
    "}\n", 
    "X = invert_features(feats, layer, model, **kwargs)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# DeepDream\n", 
    "In the summer of 2015, Google released a [blog post](http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html) describing a new method of generating images from neural networks, and they later [released code](https://github.com/google/deepdream) to generate these images.\n", 
    "\n", 
    "The idea is very simple. We pick some layer from the network, pass the starting image through the network to extract features at the chosen layer, set the gradient at that layer equal to the activations themselves, and then backpropagate to the image. This has the effect of modifying the image to amplify the activations at the chosen layer of the network.\n", 
    "\n", 
    "For DeepDream we usually extract features from one of the convolutional layers, allowing us to generate images of any resolution.\n", 
    "\n", 
    "We can implement this idea using our pretrained network. The results probably won't look as good as Google's since their network is much bigger, but we should still be able to generate some interesting images."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "def deepdream(X, layer, model, **kwargs):\n", 
    "  \"\"\"\n", 
    "  Generate a DeepDream image.\n", 
    "  \n", 
    "  Inputs:\n", 
    "  - X: Starting image, of shape (1, 3, H, W)\n", 
    "  - layer: Index of layer at which to dream\n", 
    "  - model: A PretrainedCNN object\n", 
    "  \n", 
    "  Keyword arguments:\n", 
    "  - learning_rate: How much to update the image at each iteration\n", 
    "  - max_jitter: Maximum number of pixels for jitter regularization\n", 
    "  - num_iterations: How many iterations to run for\n", 
    "  - show_every: How often to show the generated image\n", 
    "  \"\"\"\n", 
    "  \n", 
    "  X = X.copy()\n", 
    "  \n", 
    "  learning_rate = kwargs.pop('learning_rate', 5.0)\n", 
    "  max_jitter = kwargs.pop('max_jitter', 16)\n", 
    "  num_iterations = kwargs.pop('num_iterations', 100)\n", 
    "  show_every = kwargs.pop('show_every', 25)\n", 
    "  \n", 
    "  for t in xrange(num_iterations):\n", 
    "    # As a regularizer, add random jitter to the image\n", 
    "    ox, oy = np.random.randint(-max_jitter, max_jitter+1, 2)\n", 
    "    X = np.roll(np.roll(X, ox, -1), oy, -2)\n", 
    "\n", 
    "    dX = None\n", 
    "    ############################################################################\n", 
    "    # TODO: Compute the image gradient dX using the DeepDream method. You'll   #\n", 
    "    # need to use the forward and backward methods of the model object to      #\n", 
    "    # extract activations and set gradients for the chosen layer. After        #\n", 
    "    # computing the image gradient dX, you should use the learning rate to     #\n", 
    "    # update the image X.                                                      #\n", 
    "    ############################################################################\n", 
    "    pass\n", 
    "    ############################################################################\n", 
    "    #                             END OF YOUR CODE                             #\n", 
    "    ############################################################################\n", 
    "    \n", 
    "    # Undo the jitter\n", 
    "    X = np.roll(np.roll(X, -ox, -1), -oy, -2)\n", 
    "    \n", 
    "    # As a regularizer, clip the image\n", 
    "    mean_pixel = data['mean_image'].mean(axis=(1, 2), keepdims=True)\n", 
    "    X = np.clip(X, -mean_pixel, 255.0 - mean_pixel)\n", 
    "    \n", 
    "    # Periodically show the image\n", 
    "    if t == 0 or (t + 1) % show_every == 0:\n", 
    "      img = deprocess_image(X, data['mean_image'], mean='pixel')\n", 
    "      plt.imshow(img)\n", 
    "      plt.title('t = %d' % (t + 1))\n", 
    "      plt.gcf().set_size_inches(8, 8)\n", 
    "      plt.axis('off')\n", 
    "      plt.show()\n", 
    "  return X"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Generate some images!\n", 
    "Try and generate a cool-looking DeepDeam image using the pretrained network. You can try using different layers, or starting from different images. You can reduce the image size if it runs too slowly on your machine, or increase the image size if you are feeling ambitious."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "def read_image(filename, max_size):\n", 
    "  \"\"\"\n", 
    "  Read an image from disk and resize it so its larger side is max_size\n", 
    "  \"\"\"\n", 
    "  img = imread(filename)\n", 
    "  H, W, _ = img.shape\n", 
    "  if H >= W:\n", 
    "    img = imresize(img, (max_size, int(W * float(max_size) / H)))\n", 
    "  elif H < W:\n", 
    "    img = imresize(img, (int(H * float(max_size) / W), max_size))\n", 
    "  return img\n", 
    "\n", 
    "filename = 'kitten.jpg'\n", 
    "max_size = 256\n", 
    "img = read_image(filename, max_size)\n", 
    "plt.imshow(img)\n", 
    "plt.axis('off')\n", 
    "\n", 
    "# Preprocess the image by converting to float, transposing,\n", 
    "# and performing mean subtraction.\n", 
    "img_pre = preprocess_image(img, data['mean_image'], mean='pixel')\n", 
    "\n", 
    "out = deepdream(img_pre, 7, model, learning_rate=2000)"
   ], 
   "outputs": [], 
   "metadata": {
    "scrolled": false, 
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