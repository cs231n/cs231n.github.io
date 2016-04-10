{
 "nbformat_minor": 0, 
 "nbformat": 4, 
 "cells": [
  {
   "source": [
    "# Dropout\n", 
    "Dropout [1] is a technique for regularizing neural networks by randomly setting some features to zero during the forward pass. In this exercise you will implement a dropout layer and modify your fully-connected network to optionally use dropout.\n", 
    "\n", 
    "[1] Geoffrey E. Hinton et al, \"Improving neural networks by preventing co-adaptation of feature detectors\", arXiv 2012"
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
    "import time\n", 
    "import numpy as np\n", 
    "import matplotlib.pyplot as plt\n", 
    "from cs231n.classifiers.fc_net import *\n", 
    "from cs231n.data_utils import get_CIFAR10_data\n", 
    "from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array\n", 
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
    "# Dropout forward pass\n", 
    "In the file `cs231n/layers.py`, implement the forward pass for dropout. Since dropout behaves differently during training and testing, make sure to implement the operation for both modes.\n", 
    "\n", 
    "Once you have done so, run the cell below to test your implementation."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "x = np.random.randn(500, 500) + 10\n", 
    "\n", 
    "for p in [0.3, 0.6, 0.75]:\n", 
    "  out, _ = dropout_forward(x, {'mode': 'train', 'p': p})\n", 
    "  out_test, _ = dropout_forward(x, {'mode': 'test', 'p': p})\n", 
    "\n", 
    "  print 'Running tests with p = ', p\n", 
    "  print 'Mean of input: ', x.mean()\n", 
    "  print 'Mean of train-time output: ', out.mean()\n", 
    "  print 'Mean of test-time output: ', out_test.mean()\n", 
    "  print 'Fraction of train-time output set to zero: ', (out == 0).mean()\n", 
    "  print 'Fraction of test-time output set to zero: ', (out_test == 0).mean()\n", 
    "  print"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Dropout backward pass\n", 
    "In the file `cs231n/layers.py`, implement the backward pass for dropout. After doing so, run the following cell to numerically gradient-check your implementation."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "x = np.random.randn(10, 10) + 10\n", 
    "dout = np.random.randn(*x.shape)\n", 
    "\n", 
    "dropout_param = {'mode': 'train', 'p': 0.8, 'seed': 123}\n", 
    "out, cache = dropout_forward(x, dropout_param)\n", 
    "dx = dropout_backward(dout, cache)\n", 
    "dx_num = eval_numerical_gradient_array(lambda xx: dropout_forward(xx, dropout_param)[0], x, dout)\n", 
    "\n", 
    "print 'dx relative error: ', rel_error(dx, dx_num)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Fully-connected nets with Dropout\n", 
    "In the file `cs231n/classifiers/fc_net.py`, modify your implementation to use dropout. Specificially, if the constructor the the net receives a nonzero value for the `dropout` parameter, then the net should add dropout immediately after every ReLU nonlinearity. After doing so, run the following to numerically gradient-check your implementation."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, D, H1, H2, C = 2, 15, 20, 30, 10\n", 
    "X = np.random.randn(N, D)\n", 
    "y = np.random.randint(C, size=(N,))\n", 
    "\n", 
    "for dropout in [0, 0.25, 0.5]:\n", 
    "  print 'Running check with dropout = ', dropout\n", 
    "  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,\n", 
    "                            weight_scale=5e-2, dtype=np.float64,\n", 
    "                            dropout=dropout, seed=123)\n", 
    "\n", 
    "  loss, grads = model.loss(X, y)\n", 
    "  print 'Initial loss: ', loss\n", 
    "\n", 
    "  for name in sorted(grads):\n", 
    "    f = lambda _: model.loss(X, y)[0]\n", 
    "    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)\n", 
    "    print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))\n", 
    "  print"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Regularization experiment\n", 
    "As an experiment, we will train a pair of two-layer networks on 500 training examples: one will use no dropout, and one will use a dropout probability of 0.75. We will then visualize the training and validation accuracies of the two networks over time."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Train two identical nets, one with dropout and one without\n", 
    "\n", 
    "num_train = 500\n", 
    "small_data = {\n", 
    "  'X_train': data['X_train'][:num_train],\n", 
    "  'y_train': data['y_train'][:num_train],\n", 
    "  'X_val': data['X_val'],\n", 
    "  'y_val': data['y_val'],\n", 
    "}\n", 
    "\n", 
    "solvers = {}\n", 
    "dropout_choices = [0, 0.75]\n", 
    "for dropout in dropout_choices:\n", 
    "  model = FullyConnectedNet([500], dropout=dropout)\n", 
    "  print dropout\n", 
    "\n", 
    "  solver = Solver(model, small_data,\n", 
    "                  num_epochs=25, batch_size=100,\n", 
    "                  update_rule='adam',\n", 
    "                  optim_config={\n", 
    "                    'learning_rate': 5e-4,\n", 
    "                  },\n", 
    "                  verbose=True, print_every=100)\n", 
    "  solver.train()\n", 
    "  solvers[dropout] = solver"
   ], 
   "outputs": [], 
   "metadata": {
    "scrolled": false, 
    "collapsed": false
   }
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Plot train and validation accuracies of the two models\n", 
    "\n", 
    "train_accs = []\n", 
    "val_accs = []\n", 
    "for dropout in dropout_choices:\n", 
    "  solver = solvers[dropout]\n", 
    "  train_accs.append(solver.train_acc_history[-1])\n", 
    "  val_accs.append(solver.val_acc_history[-1])\n", 
    "\n", 
    "plt.subplot(3, 1, 1)\n", 
    "for dropout in dropout_choices:\n", 
    "  plt.plot(solvers[dropout].train_acc_history, 'o', label='%.2f dropout' % dropout)\n", 
    "plt.title('Train accuracy')\n", 
    "plt.xlabel('Epoch')\n", 
    "plt.ylabel('Accuracy')\n", 
    "plt.legend(ncol=2, loc='lower right')\n", 
    "  \n", 
    "plt.subplot(3, 1, 2)\n", 
    "for dropout in dropout_choices:\n", 
    "  plt.plot(solvers[dropout].val_acc_history, 'o', label='%.2f dropout' % dropout)\n", 
    "plt.title('Val accuracy')\n", 
    "plt.xlabel('Epoch')\n", 
    "plt.ylabel('Accuracy')\n", 
    "plt.legend(ncol=2, loc='lower right')\n", 
    "\n", 
    "plt.gcf().set_size_inches(15, 15)\n", 
    "plt.show()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Question\n", 
    "Explain what you see in this experiment. What does it suggest about dropout?"
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "source": [
    "# Answer\n"
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