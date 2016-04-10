{
 "nbformat_minor": 0, 
 "nbformat": 4, 
 "cells": [
  {
   "source": [
    "# Image Captioning with LSTMs\n", 
    "In the previous exercise you implemented a vanilla RNN and applied it to image captioning. In this notebook you will implement the LSTM update rule and use it for image captioning."
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
    "# Load MS-COCO data\n", 
    "As in the previous notebook, we will use the Microsoft COCO dataset for captioning."
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
    "# LSTM\n", 
    "If you read recent papers, you'll see that many people use a variant on the vanialla RNN called Long-Short Term Memory (LSTM) RNNs. Vanilla RNNs can be tough to train on long sequences due to vanishing and exploding gradiants caused by repeated matrix multiplication. LSTMs solve this problem by replacing the simple update rule of the vanilla RNN with a gating mechanism as follows.\n", 
    "\n", 
    "Similar to the vanilla RNN, at each timestep we receive an input $x_t\\in\\mathbb{R}^D$ and the previous hidden state $h_{t-1}\\in\\mathbb{R}^H$; the LSTM also maintains an $H$-dimensional *cell state*, so we also receive the previous cell state $c_{t-1}\\in\\mathbb{R}^H$. The learnable parameters of the LSTM are an *input-to-hidden* matrix $W_x\\in\\mathbb{R}^{4H\\times D}$, a *hidden-to-hidden* matrix $W_h\\in\\mathbb{R}^{4H\\times H}$ and a *bias vector* $b\\in\\mathbb{R}^{4H}$.\n", 
    "\n", 
    "At each timestep we first compute an *activation vector* $a\\in\\mathbb{R}^{4H}$ as $a=W_xx_t + W_hh_{t-1}+b$. We then divide this into four vectors $a_i,a_f,a_o,a_g\\in\\mathbb{R}^H$ where $a_i$ consists of the first $H$ elements of $a$, $a_f$ is the next $H$ elements of $a$, etc. We then compute the *input gate* $g\\in\\mathbb{R}^H$, *forget gate* $f\\in\\mathbb{R}^H$, *output gate* $o\\in\\mathbb{R}^H$ and *block input* $g\\in\\mathbb{R}^H$ as\n", 
    "\n", 
    "$$\n", 
    "\\begin{align*}\n", 
    "i = \\sigma(a_i) \\hspace{2pc}\n", 
    "f = \\sigma(a_f) \\hspace{2pc}\n", 
    "o = \\sigma(a_o) \\hspace{2pc}\n", 
    "g = \\tanh(a_g)\n", 
    "\\end{align*}\n", 
    "$$\n", 
    "\n", 
    "where $\\sigma$ is the sigmoid function and $\\tanh$ is the hyperbolic tangent, both applied elementwise.\n", 
    "\n", 
    "Finally we compute the next cell state $c_t$ and next hidden state $h_t$ as\n", 
    "\n", 
    "$$\n", 
    "c_{t} = f\\odot c_{t-1} + i\\odot g \\hspace{4pc}\n", 
    "h_t = o\\odot\\tanh(c_t)\n", 
    "$$\n", 
    "\n", 
    "where $\\odot$ is the elementwise product of vectors.\n", 
    "\n", 
    "In the rest of the notebook we will implement the LSTM update rule and apply it to the image captioning task."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "source": [
    "# LSTM: step forward\n", 
    "Implement the forward pass for a single timestep of an LSTM in the `lstm_step_forward` function in the file `cs231n/rnn_layers.py`. This should be similar to the `rnn_step_forward` function that you implemented above, but using the LSTM update rule instead.\n", 
    "\n", 
    "Once you are done, run the following to perform a simple test of your implementation. You should see errors around `1e-8` or less."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, D, H = 3, 4, 5\n", 
    "x = np.linspace(-0.4, 1.2, num=N*D).reshape(N, D)\n", 
    "prev_h = np.linspace(-0.3, 0.7, num=N*H).reshape(N, H)\n", 
    "prev_c = np.linspace(-0.4, 0.9, num=N*H).reshape(N, H)\n", 
    "Wx = np.linspace(-2.1, 1.3, num=4*D*H).reshape(D, 4 * H)\n", 
    "Wh = np.linspace(-0.7, 2.2, num=4*H*H).reshape(H, 4 * H)\n", 
    "b = np.linspace(0.3, 0.7, num=4*H)\n", 
    "\n", 
    "next_h, next_c, cache = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)\n", 
    "\n", 
    "expected_next_h = np.asarray([\n", 
    "    [ 0.24635157,  0.28610883,  0.32240467,  0.35525807,  0.38474904],\n", 
    "    [ 0.49223563,  0.55611431,  0.61507696,  0.66844003,  0.7159181 ],\n", 
    "    [ 0.56735664,  0.66310127,  0.74419266,  0.80889665,  0.858299  ]])\n", 
    "expected_next_c = np.asarray([\n", 
    "    [ 0.32986176,  0.39145139,  0.451556,    0.51014116,  0.56717407],\n", 
    "    [ 0.66382255,  0.76674007,  0.87195994,  0.97902709,  1.08751345],\n", 
    "    [ 0.74192008,  0.90592151,  1.07717006,  1.25120233,  1.42395676]])\n", 
    "\n", 
    "print 'next_h error: ', rel_error(expected_next_h, next_h)\n", 
    "print 'next_c error: ', rel_error(expected_next_c, next_c)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "#LSTM: step backward\n", 
    "Implement the backward pass for a single LSTM timestep in the function `lstm_step_backward` in the file `cs231n/rnn_layers.py`. Once you are done, run the following to perform numeric gradient checking on your implementation. You should see errors around `1e-8` or less."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, D, H = 4, 5, 6\n", 
    "x = np.random.randn(N, D)\n", 
    "prev_h = np.random.randn(N, H)\n", 
    "prev_c = np.random.randn(N, H)\n", 
    "Wx = np.random.randn(D, 4 * H)\n", 
    "Wh = np.random.randn(H, 4 * H)\n", 
    "b = np.random.randn(4 * H)\n", 
    "\n", 
    "next_h, next_c, cache = lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)\n", 
    "\n", 
    "dnext_h = np.random.randn(*next_h.shape)\n", 
    "dnext_c = np.random.randn(*next_c.shape)\n", 
    "\n", 
    "fx_h = lambda x: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]\n", 
    "fh_h = lambda h: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]\n", 
    "fc_h = lambda c: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]\n", 
    "fWx_h = lambda Wx: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]\n", 
    "fWh_h = lambda Wh: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]\n", 
    "fb_h = lambda b: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[0]\n", 
    "\n", 
    "fx_c = lambda x: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]\n", 
    "fh_c = lambda h: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]\n", 
    "fc_c = lambda c: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]\n", 
    "fWx_c = lambda Wx: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]\n", 
    "fWh_c = lambda Wh: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]\n", 
    "fb_c = lambda b: lstm_step_forward(x, prev_h, prev_c, Wx, Wh, b)[1]\n", 
    "\n", 
    "num_grad = eval_numerical_gradient_array\n", 
    "\n", 
    "dx_num = num_grad(fx_h, x, dnext_h) + num_grad(fx_c, x, dnext_c)\n", 
    "dh_num = num_grad(fh_h, prev_h, dnext_h) + num_grad(fh_c, prev_h, dnext_c)\n", 
    "dc_num = num_grad(fc_h, prev_c, dnext_h) + num_grad(fc_c, prev_c, dnext_c)\n", 
    "dWx_num = num_grad(fWx_h, Wx, dnext_h) + num_grad(fWx_c, Wx, dnext_c)\n", 
    "dWh_num = num_grad(fWh_h, Wh, dnext_h) + num_grad(fWh_c, Wh, dnext_c)\n", 
    "db_num = num_grad(fb_h, b, dnext_h) + num_grad(fb_c, b, dnext_c)\n", 
    "\n", 
    "dx, dh, dc, dWx, dWh, db = lstm_step_backward(dnext_h, dnext_c, cache)\n", 
    "\n", 
    "print 'dx error: ', rel_error(dx_num, dx)\n", 
    "print 'dh error: ', rel_error(dh_num, dh)\n", 
    "print 'dc error: ', rel_error(dc_num, dc)\n", 
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
    "# LSTM: forward\n", 
    "In the function `lstm_forward` in the file `cs231n/rnn_layers.py`, implement the `lstm_forward` function to run an LSTM forward on an entire timeseries of data.\n", 
    "\n", 
    "When you are done run the following to check your implementation. You should see an error around `1e-7`."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, D, H, T = 2, 5, 4, 3\n", 
    "x = np.linspace(-0.4, 0.6, num=N*T*D).reshape(N, T, D)\n", 
    "h0 = np.linspace(-0.4, 0.8, num=N*H).reshape(N, H)\n", 
    "Wx = np.linspace(-0.2, 0.9, num=4*D*H).reshape(D, 4 * H)\n", 
    "Wh = np.linspace(-0.3, 0.6, num=4*H*H).reshape(H, 4 * H)\n", 
    "b = np.linspace(0.2, 0.7, num=4*H)\n", 
    "\n", 
    "h, cache = lstm_forward(x, h0, Wx, Wh, b)\n", 
    "\n", 
    "expected_h = np.asarray([\n", 
    " [[ 0.01764008,  0.01823233,  0.01882671,  0.0194232 ],\n", 
    "  [ 0.11287491,  0.12146228,  0.13018446,  0.13902939],\n", 
    "  [ 0.31358768,  0.33338627,  0.35304453,  0.37250975]],\n", 
    " [[ 0.45767879,  0.4761092,   0.4936887,   0.51041945],\n", 
    "  [ 0.6704845,   0.69350089,  0.71486014,  0.7346449 ],\n", 
    "  [ 0.81733511,  0.83677871,  0.85403753,  0.86935314]]])\n", 
    "\n", 
    "print 'h error: ', rel_error(expected_h, h)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# LSTM: backward\n", 
    "Implement the backward pass for an LSTM over an entire timeseries of data in the function `lstm_backward` in the file `cs231n/rnn_layers.py`. When you are done run the following to perform numeric gradient checking on your implementation. You should see errors around `1e-8` or less."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "from cs231n.rnn_layers import lstm_forward, lstm_backward\n", 
    "\n", 
    "N, D, T, H = 2, 3, 10, 6\n", 
    "\n", 
    "x = np.random.randn(N, T, D)\n", 
    "h0 = np.random.randn(N, H)\n", 
    "Wx = np.random.randn(D, 4 * H)\n", 
    "Wh = np.random.randn(H, 4 * H)\n", 
    "b = np.random.randn(4 * H)\n", 
    "\n", 
    "out, cache = lstm_forward(x, h0, Wx, Wh, b)\n", 
    "\n", 
    "dout = np.random.randn(*out.shape)\n", 
    "\n", 
    "dx, dh0, dWx, dWh, db = lstm_backward(dout, cache)\n", 
    "\n", 
    "fx = lambda x: lstm_forward(x, h0, Wx, Wh, b)[0]\n", 
    "fh0 = lambda h0: lstm_forward(x, h0, Wx, Wh, b)[0]\n", 
    "fWx = lambda Wx: lstm_forward(x, h0, Wx, Wh, b)[0]\n", 
    "fWh = lambda Wh: lstm_forward(x, h0, Wx, Wh, b)[0]\n", 
    "fb = lambda b: lstm_forward(x, h0, Wx, Wh, b)[0]\n", 
    "\n", 
    "dx_num = eval_numerical_gradient_array(fx, x, dout)\n", 
    "dh0_num = eval_numerical_gradient_array(fh0, h0, dout)\n", 
    "dWx_num = eval_numerical_gradient_array(fWx, Wx, dout)\n", 
    "dWh_num = eval_numerical_gradient_array(fWh, Wh, dout)\n", 
    "db_num = eval_numerical_gradient_array(fb, b, dout)\n", 
    "\n", 
    "print 'dx error: ', rel_error(dx_num, dx)\n", 
    "print 'dh0 error: ', rel_error(dx_num, dx)\n", 
    "print 'dWx error: ', rel_error(dx_num, dx)\n", 
    "print 'dWh error: ', rel_error(dx_num, dx)\n", 
    "print 'db error: ', rel_error(dx_num, dx)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "#LSTM captioning model\n", 
    "Now that you have implemented an LSTM, update the implementation of the `loss` method of the `CaptioningRNN` class in the file `cs231n/classifiers/rnn.py` to handle the case where `self.cell_type` is `lstm`. This should require adding less than 10 lines of code.\n", 
    "\n", 
    "Once you have done so, run the following to check your implementation. You should see a difference of less than `1e-10`."
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
    "          cell_type='lstm',\n", 
    "          dtype=np.float64)\n", 
    "\n", 
    "# Set all model parameters to fixed values\n", 
    "for k, v in model.params.iteritems():\n", 
    "  model.params[k] = np.linspace(-1.4, 1.3, num=v.size).reshape(*v.shape)\n", 
    "\n", 
    "features = np.linspace(-0.5, 1.7, num=N*D).reshape(N, D)\n", 
    "captions = (np.arange(N * T) % V).reshape(N, T)\n", 
    "\n", 
    "loss, grads = model.loss(features, captions)\n", 
    "expected_loss = 9.82445935443\n", 
    "\n", 
    "print 'loss: ', loss\n", 
    "print 'expected loss: ', expected_loss\n", 
    "print 'difference: ', abs(loss - expected_loss)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Overfit LSTM captioning model\n", 
    "Run the following to overfit an LSTM captioning model on the same small dataset as we used for the RNN above."
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
    "small_lstm_model = CaptioningRNN(\n", 
    "          cell_type='lstm',\n", 
    "          word_to_idx=data['word_to_idx'],\n", 
    "          input_dim=data['train_features'].shape[1],\n", 
    "          hidden_dim=512,\n", 
    "          wordvec_dim=256,\n", 
    "          dtype=np.float32,\n", 
    "        )\n", 
    "\n", 
    "small_lstm_solver = CaptioningSolver(small_lstm_model, small_data,\n", 
    "           update_rule='adam',\n", 
    "           num_epochs=50,\n", 
    "           batch_size=25,\n", 
    "           optim_config={\n", 
    "             'learning_rate': 5e-3,\n", 
    "           },\n", 
    "           lr_decay=0.995,\n", 
    "           verbose=True, print_every=10,\n", 
    "         )\n", 
    "\n", 
    "small_lstm_solver.train()\n", 
    "\n", 
    "# Plot the training losses\n", 
    "plt.plot(small_lstm_solver.loss_history)\n", 
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
    "# LSTM test-time sampling\n", 
    "Modify the `sample` method of the `CaptioningRNN` class to handle the case where `self.cell_type` is `lstm`. This should take fewer than 10 lines of code.\n", 
    "\n", 
    "When you are done run the following to sample from your overfit LSTM model on some training and validation set samples."
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
    "  sample_captions = small_lstm_model.sample(features)\n", 
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
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Train a good captioning model!\n", 
    "Using the pieces you have implemented in this and the previous notebook, try to train a captioning model that gives decent qualitative results (better than the random garbage you saw with the overfit models) when sampling on the validation set. You can subsample the training set if you want; we just want to see samples on the validatation set that are better than random.\n", 
    "\n", 
    "Don't spend too much time on this part; we don't have any explicit accuracy thresholds you need to meet."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "pass\n"
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
    "pass\n"
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