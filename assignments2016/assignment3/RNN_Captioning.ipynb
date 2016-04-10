{
 "nbformat_minor": 0, 
 "nbformat": 4, 
 "cells": [
  {
   "source": [
    "# Image Captioning with RNNs\n", 
    "In this exercise you will implement a vanilla recurrent neural networks and use them it to train a model that can generate novel captions for images."
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
    "import matplotlib.pyplot as plt\n", 
    "\n", 
    "from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array\n", 
    "from cs231n.rnn_layers import *\n", 
    "from cs231n.captioning_solver import CaptioningSolver\n", 
    "from cs231n.classifiers.rnn import CaptioningRNN\n", 
    "from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions\n", 
    "from cs231n.image_utils import image_from_url\n", 
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
   "source": [
    "# Microsoft COCO\n", 
    "For this exercise we will use the 2014 release of the [Microsoft COCO dataset](http://mscoco.org/) which has become the standard testbed for image captioning. The dataset consists of 80,000 training images and 40,000 validation images, each annotated with 5 captions written by workers on Amazon Mechanical Turk.\n", 
    "\n", 
    "To download the data, change to the `cs231n/datasets` directory and run the script `get_coco_captioning.sh`.\n", 
    "\n", 
    "We have preprocessed the data and extracted features for you already. For all images we have extracted features from the fc7 layer of the VGG-16 network pretrained on ImageNet; these features are stored in the files `train2014_vgg16_fc7.h5` and `val2014_vgg16_fc7.h5` respectively. To cut down on processing time and memory requirements, we have reduced the dimensionality of the features from 4096 to 512; these features can be found in the files `train2014_vgg16_fc7_pca.h5` and `val2014_vgg16_fc7_pca.h5`.\n", 
    "\n", 
    "The raw images take up a lot of space (nearly 20GB) so we have not included them in the download. However all images are taken from Flickr, and URLs of the training and validation images are stored in the files `train2014_urls.txt` and `val2014_urls.txt` respectively. This allows you to download images on the fly for visualization. Since images are downloaded on-the-fly, **you must be connected to the internet to view images**.\n", 
    "\n", 
    "Dealing with strings is inefficient, so we will work with an encoded version of the captions. Each word is assigned an integer ID, allowing us to represent a caption by a sequence of integers. The mapping between integer IDs and words is in the file `coco2014_vocab.json`, and you can use the function `decode_captions` from the file `cs231n/coco_utils.py` to convert numpy arrays of integer IDs back into strings.\n", 
    "\n", 
    "There are a couple special tokens that we add to the vocabulary. We prepend a special `<START>` token and append an `<END>` token to the beginning and end of each caption respectively. Rare words are replaced with a special `<UNK>` token (for \"unknown\"). In addition, since we want to train with minibatches containing captions of different lengths, we pad short captions with a special `<NULL>` token after the `<END>` token and don't compute loss or gradient for `<NULL>` tokens. Since they are a bit of a pain, we have taken care of all implementation details around special tokens for you.\n", 
    "\n", 
    "You can load all of the MS-COCO data (captions, features, URLs, and vocabulary) using the `load_coco_data` function from the file `cs231n/coco_utils.py`. Run the following cell to do so:"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Load COCO data from disk; this returns a dictionary\n", 
    "# We'll work with dimensionality-reduced features for this notebook, but feel\n", 
    "# free to experiment with the original features by changing the flag below.\n", 
    "data = load_coco_data(pca_features=True)\n", 
    "\n", 
    "# Print out all the keys and values from the data dictionary\n", 
    "for k, v in data.iteritems():\n", 
    "  if type(v) == np.ndarray:\n", 
    "    print k, type(v), v.shape, v.dtype\n", 
    "  else:\n", 
    "    print k, type(v), len(v)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "## Look at the data\n", 
    "It is always a good idea to look at examples from the dataset before working with it.\n", 
    "\n", 
    "You can use the `sample_coco_minibatch` function from the file `cs231n/coco_utils.py` to sample minibatches of data from the data structure returned from `load_coco_data`. Run the following to sample a small minibatch of training data and show the images and their captions. Running it multiple times and looking at the results helps you to get a sense of the dataset.\n", 
    "\n", 
    "Note that we decode the captions using the `decode_captions` function and that we download the images on-the-fly using their Flickr URL, so **you must be connected to the internet to viw images**."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Sample a minibatch and show the images and captions\n", 
    "batch_size = 3\n", 
    "\n", 
    "captions, features, urls = sample_coco_minibatch(data, batch_size=batch_size)\n", 
    "for i, (caption, url) in enumerate(zip(captions, urls)):\n", 
    "  plt.imshow(image_from_url(url))\n", 
    "  plt.axis('off')\n", 
    "  caption_str = decode_captions(caption, data['idx_to_word'])\n", 
    "  plt.title(caption_str)\n", 
    "  plt.show()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Recurrent Neural Networks\n", 
    "As discussed in lecture, we will use recurrent neural network (RNN) language models for image captioning. The file `cs231n/rnn_layers.py` contains implementations of different layer types that are needed for recurrent neural networks, and the file `cs231n/classifiers/rnn.py` uses these layers to implement an image captioning model.\n", 
    "\n", 
    "We will first implement different types of RNN layers in `cs231n/rnn_layers.py`."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "source": [
    "# Vanilla RNN: step forward\n", 
    "Open the file `cs231n/rnn_layers.py`. This file implements the forward and backward passes for different types of layers that are commonly used in recurrent neural networks.\n", 
    "\n", 
    "First implement the function `rnn_step_forward` which implements the forward pass for a single timestep of a vanilla recurrent neural network. After doing so run the following to check your implementation."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, D, H = 3, 10, 4\n", 
    "\n", 
    "x = np.linspace(-0.4, 0.7, num=N*D).reshape(N, D)\n", 
    "prev_h = np.linspace(-0.2, 0.5, num=N*H).reshape(N, H)\n", 
    "Wx = np.linspace(-0.1, 0.9, num=D*H).reshape(D, H)\n", 
    "Wh = np.linspace(-0.3, 0.7, num=H*H).reshape(H, H)\n", 
    "b = np.linspace(-0.2, 0.4, num=H)\n", 
    "\n", 
    "next_h, _ = rnn_step_forward(x, prev_h, Wx, Wh, b)\n", 
    "expected_next_h = np.asarray([\n", 
    "  [-0.58172089, -0.50182032, -0.41232771, -0.31410098],\n", 
    "  [ 0.66854692,  0.79562378,  0.87755553,  0.92795967],\n", 
    "  [ 0.97934501,  0.99144213,  0.99646691,  0.99854353]])\n", 
    "\n", 
    "print 'next_h error: ', rel_error(expected_next_h, next_h)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Vanilla RNN: step backward\n", 
    "In the file `cs231n/rnn_layers.py` implement the `rnn_step_backward` function. After doing so run the following to numerically gradient check your implementation. You should see errors less than `1e-8`."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "from cs231n.rnn_layers import rnn_step_forward, rnn_step_backward\n", 
    "\n", 
    "N, D, H = 4, 5, 6\n", 
    "x = np.random.randn(N, D)\n", 
    "h = np.random.randn(N, H)\n", 
    "Wx = np.random.randn(D, H)\n", 
    "Wh = np.random.randn(H, H)\n", 
    "b = np.random.randn(H)\n", 
    "\n", 
    "out, cache = rnn_step_forward(x, h, Wx, Wh, b)\n", 
    "\n", 
    "dnext_h = np.random.randn(*out.shape)\n", 
    "\n", 
    "fx = lambda x: rnn_step_forward(x, h, Wx, Wh, b)[0]\n", 
    "fh = lambda prev_h: rnn_step_forward(x, h, Wx, Wh, b)[0]\n", 
    "fWx = lambda Wx: rnn_step_forward(x, h, Wx, Wh, b)[0]\n", 
    "fWh = lambda Wh: rnn_step_forward(x, h, Wx, Wh, b)[0]\n", 
    "fb = lambda b: rnn_step_forward(x, h, Wx, Wh, b)[0]\n", 
    "\n", 
    "dx_num = eval_numerical_gradient_array(fx, x, dnext_h)\n", 
    "dprev_h_num = eval_numerical_gradient_array(fh, h, dnext_h)\n", 
    "dWx_num = eval_numerical_gradient_array(fWx, Wx, dnext_h)\n", 
    "dWh_num = eval_numerical_gradient_array(fWh, Wh, dnext_h)\n", 
    "db_num = eval_numerical_gradient_array(fb, b, dnext_h)\n", 
    "\n", 
    "dx, dprev_h, dWx, dWh, db = rnn_step_backward(dnext_h, cache)\n", 
    "\n", 
    "print 'dx error: ', rel_error(dx_num, dx)\n", 
    "print 'dprev_h error: ', rel_error(dprev_h_num, dprev_h)\n", 
    "print 'dWx error: ', rel_error(dWx_num, dWx)\n", 
    "print 'dWh error: ', rel_error(dWh_num, dWh)\n", 
    "print 'db error: ', rel_error(db_num, db)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Vanilla RNN: forward\n", 
    "Now that you have implemented the forward and backward passes for a single timestep of a vanilla RNN, you will combine these pieces to implement a RNN that process an entire sequence of data.\n", 
    "\n", 
    "In the file `cs231n/rnn_layers.py`, implement the function `rnn_forward`. This should be implemented using the `rnn_step_forward` function that you defined above. After doing so run the following to check your implementation. You should see errors less than `1e-7`."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, T, D, H = 2, 3, 4, 5\n", 
    "\n", 
    "x = np.linspace(-0.1, 0.3, num=N*T*D).reshape(N, T, D)\n", 
    "h0 = np.linspace(-0.3, 0.1, num=N*H).reshape(N, H)\n", 
    "Wx = np.linspace(-0.2, 0.4, num=D*H).reshape(D, H)\n", 
    "Wh = np.linspace(-0.4, 0.1, num=H*H).reshape(H, H)\n", 
    "b = np.linspace(-0.7, 0.1, num=H)\n", 
    "\n", 
    "h, _ = rnn_forward(x, h0, Wx, Wh, b)\n", 
    "expected_h = np.asarray([\n", 
    "  [\n", 
    "    [-0.42070749, -0.27279261, -0.11074945,  0.05740409,  0.22236251],\n", 
    "    [-0.39525808, -0.22554661, -0.0409454,   0.14649412,  0.32397316],\n", 
    "    [-0.42305111, -0.24223728, -0.04287027,  0.15997045,  0.35014525],\n", 
    "  ],\n", 
    "  [\n", 
    "    [-0.55857474, -0.39065825, -0.19198182,  0.02378408,  0.23735671],\n", 
    "    [-0.27150199, -0.07088804,  0.13562939,  0.33099728,  0.50158768],\n", 
    "    [-0.51014825, -0.30524429, -0.06755202,  0.17806392,  0.40333043]]])\n", 
    "print 'h error: ', rel_error(expected_h, h)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Vanilla RNN: backward\n", 
    "In the file `cs231n/rnn_layers.py`, implement the backward pass for a vanilla RNN in the function `rnn_backward`. This should run back-propagation over the entire sequence, calling into the `rnn_step_backward` function that you defined above."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, D, T, H = 2, 3, 10, 5\n", 
    "\n", 
    "x = np.random.randn(N, T, D)\n", 
    "h0 = np.random.randn(N, H)\n", 
    "Wx = np.random.randn(D, H)\n", 
    "Wh = np.random.randn(H, H)\n", 
    "b = np.random.randn(H)\n", 
    "\n", 
    "out, cache = rnn_forward(x, h0, Wx, Wh, b)\n", 
    "\n", 
    "dout = np.random.randn(*out.shape)\n", 
    "\n", 
    "dx, dh0, dWx, dWh, db = rnn_backward(dout, cache)\n", 
    "\n", 
    "fx = lambda x: rnn_forward(x, h0, Wx, Wh, b)[0]\n", 
    "fh0 = lambda h0: rnn_forward(x, h0, Wx, Wh, b)[0]\n", 
    "fWx = lambda Wx: rnn_forward(x, h0, Wx, Wh, b)[0]\n", 
    "fWh = lambda Wh: rnn_forward(x, h0, Wx, Wh, b)[0]\n", 
    "fb = lambda b: rnn_forward(x, h0, Wx, Wh, b)[0]\n", 
    "\n", 
    "dx_num = eval_numerical_gradient_array(fx, x, dout)\n", 
    "dh0_num = eval_numerical_gradient_array(fh0, h0, dout)\n", 
    "dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)\n", 
    "dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)\n", 
    "db_num = eval_numerical_gradient_array(fb, b, dout)\n", 
    "\n", 
    "print 'dx error: ', rel_error(dx_num, dx)\n", 
    "print 'dh0 error: ', rel_error(dh0_num, dh0)\n", 
    "print 'dWx error: ', rel_error(dWx_num, dWx)\n", 
    "print 'dWh error: ', rel_error(dWh_num, dWh)\n", 
    "print 'db error: ', rel_error(db_num, db)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Word embedding: forward\n", 
    "In deep learning systems, we commonly represent words using vectors. Each word of the vocabulary will be associated with a vector, and these vectors will be learned jointly with the rest of the system.\n", 
    "\n", 
    "In the file `cs231n/rnn_layers.py`, implement the function `word_embedding_forward` to convert words (represented by integers) into vectors. Run the following to check your implementation. You should see error around `1e-8`."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, T, V, D = 2, 4, 5, 3\n", 
    "\n", 
    "x = np.asarray([[0, 3, 1, 2], [2, 1, 0, 3]])\n", 
    "W = np.linspace(0, 1, num=V*D).reshape(V, D)\n", 
    "\n", 
    "out, _ = word_embedding_forward(x, W)\n", 
    "expected_out = np.asarray([\n", 
    " [[ 0.,          0.07142857,  0.14285714],\n", 
    "  [ 0.64285714,  0.71428571,  0.78571429],\n", 
    "  [ 0.21428571,  0.28571429,  0.35714286],\n", 
    "  [ 0.42857143,  0.5,         0.57142857]],\n", 
    " [[ 0.42857143,  0.5,         0.57142857],\n", 
    "  [ 0.21428571,  0.28571429,  0.35714286],\n", 
    "  [ 0.,          0.07142857,  0.14285714],\n", 
    "  [ 0.64285714,  0.71428571,  0.78571429]]])\n", 
    "\n", 
    "print 'out error: ', rel_error(expected_out, out)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Word embedding: backward\n", 
    "Implement the backward pass for the word embedding function in the function `word_embedding_backward`. After doing so run the following to numerically gradient check your implementation. You should see errors less than `1e-11`."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, T, V, D = 50, 3, 5, 6\n", 
    "\n", 
    "x = np.random.randint(V, size=(N, T))\n", 
    "W = np.random.randn(V, D)\n", 
    "\n", 
    "out, cache = word_embedding_forward(x, W)\n", 
    "dout = np.random.randn(*out.shape)\n", 
    "dW = word_embedding_backward(dout, cache)\n", 
    "\n", 
    "f = lambda W: word_embedding_forward(x, W)[0]\n", 
    "dW_num = eval_numerical_gradient_array(f, W, dout)\n", 
    "\n", 
    "print 'dW error: ', rel_error(dW, dW_num)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Temporal Affine layer\n", 
    "At every timestep we use an affine function to transform the RNN hidden vector at that timestep into scores for each word in the vocabulary. Because this is very similar to the affine layer that you implemented in assignment 2, we have provided this function for you in the `temporal_affine_forward` and `temporal_affine_backward` functions in the file `cs231n/rnn_layers.py`. Run the following to perform numeric gradient checking on the implementation."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Gradient check for temporal affine layer\n", 
    "N, T, D, M = 2, 3, 4, 5\n", 
    "\n", 
    "x = np.random.randn(N, T, D)\n", 
    "w = np.random.randn(D, M)\n", 
    "b = np.random.randn(M)\n", 
    "\n", 
    "out, cache = temporal_affine_forward(x, w, b)\n", 
    "\n", 
    "dout = np.random.randn(*out.shape)\n", 
    "\n", 
    "fx = lambda x: temporal_affine_forward(x, w, b)[0]\n", 
    "fw = lambda w: temporal_affine_forward(x, w, b)[0]\n", 
    "fb = lambda b: temporal_affine_forward(x, w, b)[0]\n", 
    "\n", 
    "dx_num = eval_numerical_gradient_array(fx, x, dout)\n", 
    "dw_num = eval_numerical_gradient_array(fw, w, dout)\n", 
    "db_num = eval_numerical_gradient_array(fb, b, dout)\n", 
    "\n", 
    "dx, dw, db = temporal_affine_backward(dout, cache)\n", 
    "\n", 
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
    "# Temporal Softmax loss\n", 
    "In an RNN language model, at every timestep we produce a score for each word in the vocabulary. We know the ground-truth word at each timestep, so we use a softmax loss function to compute loss and gradient at each timestep. We sum the losses over time and average them over the minibatch.\n", 
    "\n", 
    "However there is one wrinke: since we operate over minibatches and different captions may have different lengths, we append `<NULL>` tokens to the end of each caption so they all have the same length. We don't want these `<NULL>` tokens to count toward the loss or gradient, so in addition to scores and ground-truth labels our loss function also accepts a `mask` array that tells it which elements of the scores count towards the loss.\n", 
    "\n", 
    "Since this is very similar to the softmax loss function you implemented in assignment 1, we have implemented this loss function for you; look at the `temporal_softmax_loss` function in the file `cs231n/rnn_layers.py`.\n", 
    "\n", 
    "Run the following cell to sanity check the loss and perform numeric gradient checking on the function."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Sanity check for temporal softmax loss\n", 
    "from cs231n.rnn_layers import temporal_softmax_loss\n", 
    "\n", 
    "N, T, V = 100, 1, 10\n", 
    "\n", 
    "def check_loss(N, T, V, p):\n", 
    "  x = 0.001 * np.random.randn(N, T, V)\n", 
    "  y = np.random.randint(V, size=(N, T))\n", 
    "  mask = np.random.rand(N, T) <= p\n", 
    "  print temporal_softmax_loss(x, y, mask)[0]\n", 
    "  \n", 
    "check_loss(100, 1, 10, 1.0)   # Should be about 2.3\n", 
    "check_loss(100, 10, 10, 1.0)  # Should be about 23\n", 
    "check_loss(5000, 10, 10, 0.1) # Should be about 2.3\n", 
    "\n", 
    "# Gradient check for temporal softmax loss\n", 
    "N, T, V = 7, 8, 9\n", 
    "\n", 
    "x = np.random.randn(N, T, V)\n", 
    "y = np.random.randint(V, size=(N, T))\n", 
    "mask = (np.random.rand(N, T) > 0.5)\n", 
    "\n", 
    "loss, dx = temporal_softmax_loss(x, y, mask, verbose=False)\n", 
    "\n", 
    "dx_num = eval_numerical_gradient(lambda x: temporal_softmax_loss(x, y, mask)[0], x, verbose=False)\n", 
    "\n", 
    "print 'dx error: ', rel_error(dx, dx_num)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# RNN for image captioning\n", 
    "Now that you have implemented the necessary layers, you can combine them to build an image captioning model. Open the file `cs231n/classifiers/rnn.py` and look at the `CaptioningRNN` class.\n", 
    "\n", 
    "Implement the forward and backward pass of the model in the `loss` function. For now you only need to implement the case where `cell_type='rnn'` for vanialla RNNs; you will implement the LSTM case later. After doing so, run the following to check your forward pass using a small test case; you should see error less than `1e-10`."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, D, W, H = 10, 20, 30, 40\n", 
    "word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}\n", 
    "V = len(word_to_idx)\n", 
    "T = 13\n", 
    "\n", 
    "model = CaptioningRNN(word_to_idx,\n", 
    "          input_dim=D,\n", 
    "          wordvec_dim=W,\n", 
    "          hidden_dim=H,\n", 
    "          cell_type='rnn',\n", 
    "          dtype=np.float64)\n", 
    "\n", 
    "# Set all model parameters to fixed values\n", 
    "for k, v in model.params.iteritems():\n", 
    "  model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)\n", 
    "\n", 
    "features = np.linspace(-1.5, 0.3, num=(N * D)).reshape(N, D)\n", 
    "captions = (np.arange(N * T) % V).reshape(N, T)\n", 
    "\n", 
    "loss, grads = model.loss(features, captions)\n", 
    "expected_loss = 9.83235591003\n", 
    "\n", 
    "print 'loss: ', loss\n", 
    "print 'expected loss: ', expected_loss\n", 
    "print 'difference: ', abs(loss - expected_loss)"
   ], 
   "outputs": [], 
   "metadata": {
    "scrolled": false, 
    "collapsed": false
   }
  }, 
  {
   "source": [
    "Run the following cell to perform numeric gradient checking on the `CaptioningRNN` class; you should errors around `1e-7` or less."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "batch_size = 2\n", 
    "timesteps = 3\n", 
    "input_dim = 4\n", 
    "wordvec_dim = 5\n", 
    "hidden_dim = 6\n", 
    "word_to_idx = {'<NULL>': 0, 'cat': 2, 'dog': 3}\n", 
    "vocab_size = len(word_to_idx)\n", 
    "\n", 
    "captions = np.random.randint(vocab_size, size=(batch_size, timesteps))\n", 
    "features = np.random.randn(batch_size, input_dim)\n", 
    "\n", 
    "model = CaptioningRNN(word_to_idx,\n", 
    "          input_dim=input_dim,\n", 
    "          wordvec_dim=wordvec_dim,\n", 
    "          hidden_dim=hidden_dim,\n", 
    "          cell_type='rnn',\n", 
    "          dtype=np.float64,\n", 
    "        )\n", 
    "\n", 
    "loss, grads = model.loss(features, captions)\n", 
    "\n", 
    "for param_name in sorted(grads):\n", 
    "  f = lambda _: model.loss(features, captions)[0]\n", 
    "  param_grad_num = eval_numerical_gradient(f, model.params[param_name], verbose=False, h=1e-6)\n", 
    "  e = rel_error(param_grad_num, grads[param_name])\n", 
    "  print '%s relative error: %e' % (param_name, e)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Overfit small data\n", 
    "Similar to the `Solver` class that we used to train image classification models on the previous assignment, on this assignment we use a `CaptioningSolver` class to train image captioning models. Open the file `cs231n/captioning_solver.py` and read through the `CaptioningSolver` class; it should look very familiar.\n", 
    "\n", 
    "Once you have familiarized yourself with the API, run the following to make sure your model overfit a small sample of 100 training examples. You should see losses around 1."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "small_data = load_coco_data(max_train=50)\n", 
    "\n", 
    "small_rnn_model = CaptioningRNN(\n", 
    "          cell_type='rnn',\n", 
    "          word_to_idx=data['word_to_idx'],\n", 
    "          input_dim=data['train_features'].shape[1],\n", 
    "          hidden_dim=512,\n", 
    "          wordvec_dim=256,\n", 
    "        )\n", 
    "\n", 
    "small_rnn_solver = CaptioningSolver(small_rnn_model, small_data,\n", 
    "           update_rule='adam',\n", 
    "           num_epochs=50,\n", 
    "           batch_size=25,\n", 
    "           optim_config={\n", 
    "             'learning_rate': 5e-3,\n", 
    "           },\n", 
    "           lr_decay=0.95,\n", 
    "           verbose=True, print_every=10,\n", 
    "         )\n", 
    "\n", 
    "small_rnn_solver.train()\n", 
    "\n", 
    "# Plot the training losses\n", 
    "plt.plot(small_rnn_solver.loss_history)\n", 
    "plt.xlabel('Iteration')\n", 
    "plt.ylabel('Loss')\n", 
    "plt.title('Training loss history')\n", 
    "plt.show()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Test-time sampling\n", 
    "Unlike classification models, image captioning models behave very differently at training time and at test time. At training time, we have access to the ground-truth caption so we feed ground-truth words as input to the RNN at each timestep. At test time, we sample from the distribution over the vocabulary at each timestep, and feed the sample as input to the RNN at the next timestep.\n", 
    "\n", 
    "In the file `cs231n/classifiers/rnn.py`, implement the `sample` method for test-time sampling. After doing so, run the following to sample from your overfit model on both training and validation data. The samples on training data should be very good; the samples on validation data probably won't make sense."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "for split in ['train', 'val']:\n", 
    "  minibatch = sample_coco_minibatch(small_data, split=split, batch_size=2)\n", 
    "  gt_captions, features, urls = minibatch\n", 
    "  gt_captions = decode_captions(gt_captions, data['idx_to_word'])\n", 
    "\n", 
    "  sample_captions = small_rnn_model.sample(features)\n", 
    "  sample_captions = decode_captions(sample_captions, data['idx_to_word'])\n", 
    "\n", 
    "  for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):\n", 
    "    plt.imshow(image_from_url(url))\n", 
    "    plt.title('%s\\n%s\\nGT:%s' % (split, sample_caption, gt_caption))\n", 
    "    plt.axis('off')\n", 
    "    plt.show()"
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