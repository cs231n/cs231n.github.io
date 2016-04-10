{
 "nbformat_minor": 0, 
 "nbformat": 4, 
 "cells": [
  {
   "source": [
    "# Batch Normalization\n", 
    "One way to make deep networks easier to train is to use more sophisticated optimization procedures such as SGD+momentum, RMSProp, or Adam. Another strategy is to change the architecture of the network to make it easier to train. One idea along these lines is batch normalization which was recently proposed by [3].\n", 
    "\n", 
    "The idea is relatively straightforward. Machine learning methods tend to work better when their input data consists of uncorrelated features with zero mean and unit variance. When training a neural network, we can preprocess the data before feeding it to the network to explicitly decorrelate its features; this will ensure that the first layer of the network sees data that follows a nice distribution. However even if we preprocess the input data, the activations at deeper layers of the network will likely no longer be decorrelated and will no longer have zero mean or unit variance since they are output from earlier layers in the network. Even worse, during the training process the distribution of features at each layer of the network will shift as the weights of each layer are updated.\n", 
    "\n", 
    "The authors of [3] hypothesize that the shifting distribution of features inside deep neural networks may make training deep networks more difficult. To overcome this problem, [3] proposes to insert batch normalization layers into the network. At training time, a batch normalization layer uses a minibatch of data to estimate the mean and standard deviation of each feature. These estimated means and standard deviations are then used to center and normalize the features of the minibatch. A running average of these means and standard deviations is kept during training, and at test time these running averages are used to center and normalize features.\n", 
    "\n", 
    "It is possible that this normalization strategy could reduce the representational power of the network, since it may sometimes be optimal for certain layers to have features that are not zero-mean or unit variance. To this end, the batch normalization layer includes learnable shift and scale parameters for each feature dimension.\n", 
    "\n", 
    "[3] Sergey Ioffe and Christian Szegedy, \"Batch Normalization: Accelerating Deep Network Training by Reducing\n", 
    "Internal Covariate Shift\", ICML 2015."
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
    "## Batch normalization: Forward\n", 
    "In the file `cs231n/layers.py`, implement the batch normalization forward pass in the function `batchnorm_forward`. Once you have done so, run the following to test your implementation."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Check the training-time forward pass by checking means and variances\n", 
    "# of features both before and after batch normalization\n", 
    "\n", 
    "# Simulate the forward pass for a two-layer network\n", 
    "N, D1, D2, D3 = 200, 50, 60, 3\n", 
    "X = np.random.randn(N, D1)\n", 
    "W1 = np.random.randn(D1, D2)\n", 
    "W2 = np.random.randn(D2, D3)\n", 
    "a = np.maximum(0, X.dot(W1)).dot(W2)\n", 
    "\n", 
    "print 'Before batch normalization:'\n", 
    "print '  means: ', a.mean(axis=0)\n", 
    "print '  stds: ', a.std(axis=0)\n", 
    "\n", 
    "# Means should be close to zero and stds close to one\n", 
    "print 'After batch normalization (gamma=1, beta=0)'\n", 
    "a_norm, _ = batchnorm_forward(a, np.ones(D3), np.zeros(D3), {'mode': 'train'})\n", 
    "print '  mean: ', a_norm.mean(axis=0)\n", 
    "print '  std: ', a_norm.std(axis=0)\n", 
    "\n", 
    "# Now means should be close to beta and stds close to gamma\n", 
    "gamma = np.asarray([1.0, 2.0, 3.0])\n", 
    "beta = np.asarray([11.0, 12.0, 13.0])\n", 
    "a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})\n", 
    "print 'After batch normalization (nontrivial gamma, beta)'\n", 
    "print '  means: ', a_norm.mean(axis=0)\n", 
    "print '  stds: ', a_norm.std(axis=0)"
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
    "N, D1, D2, D3 = 200, 50, 60, 3\n", 
    "W1 = np.random.randn(D1, D2)\n", 
    "W2 = np.random.randn(D2, D3)\n", 
    "\n", 
    "bn_param = {'mode': 'train'}\n", 
    "gamma = np.ones(D3)\n", 
    "beta = np.zeros(D3)\n", 
    "for t in xrange(50):\n", 
    "  X = np.random.randn(N, D1)\n", 
    "  a = np.maximum(0, X.dot(W1)).dot(W2)\n", 
    "  batchnorm_forward(a, gamma, beta, bn_param)\n", 
    "bn_param['mode'] = 'test'\n", 
    "X = np.random.randn(N, D1)\n", 
    "a = np.maximum(0, X.dot(W1)).dot(W2)\n", 
    "a_norm, _ = batchnorm_forward(a, gamma, beta, bn_param)\n", 
    "\n", 
    "# Means should be close to zero and stds close to one, but will be\n", 
    "# noisier than training-time forward passes.\n", 
    "print 'After batch normalization (test-time):'\n", 
    "print '  means: ', a_norm.mean(axis=0)\n", 
    "print '  stds: ', a_norm.std(axis=0)"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "## Batch Normalization: backward\n", 
    "Now implement the backward pass for batch normalization in the function `batchnorm_backward`.\n", 
    "\n", 
    "To derive the backward pass you should write out the computation graph for batch normalization and backprop through each of the intermediate nodes. Some intermediates may have multiple outgoing branches; make sure to sum gradients across these branches in the backward pass.\n", 
    "\n", 
    "Once you have finished, run the following to numerically check your backward pass."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Gradient check batchnorm backward pass\n", 
    "\n", 
    "N, D = 4, 5\n", 
    "x = 5 * np.random.randn(N, D) + 12\n", 
    "gamma = np.random.randn(D)\n", 
    "beta = np.random.randn(D)\n", 
    "dout = np.random.randn(N, D)\n", 
    "\n", 
    "bn_param = {'mode': 'train'}\n", 
    "fx = lambda x: batchnorm_forward(x, gamma, beta, bn_param)[0]\n", 
    "fg = lambda a: batchnorm_forward(x, gamma, beta, bn_param)[0]\n", 
    "fb = lambda b: batchnorm_forward(x, gamma, beta, bn_param)[0]\n", 
    "\n", 
    "dx_num = eval_numerical_gradient_array(fx, x, dout)\n", 
    "da_num = eval_numerical_gradient_array(fg, gamma, dout)\n", 
    "db_num = eval_numerical_gradient_array(fb, beta, dout)\n", 
    "\n", 
    "_, cache = batchnorm_forward(x, gamma, beta, bn_param)\n", 
    "dx, dgamma, dbeta = batchnorm_backward(dout, cache)\n", 
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
    "## Batch Normalization: alternative backward\n", 
    "In class we talked about two different implementations for the sigmoid backward pass. One strategy is to write out a computation graph composed of simple operations and backprop through all intermediate values. Another strategy is to work out the derivatives on paper. For the sigmoid function, it turns out that you can derive a very simple formula for the backward pass by simplifying gradients on paper.\n", 
    "\n", 
    "Surprisingly, it turns out that you can also derive a simple expression for the batch normalization backward pass if you work out derivatives on paper and simplify. After doing so, implement the simplified batch normalization backward pass in the function `batchnorm_backward_alt` and compare the two implementations by running the following. Your two implementations should compute nearly identical results, but the alternative implementation should be a bit faster.\n", 
    "\n", 
    "NOTE: You can still complete the rest of the assignment if you don't figure this part out, so don't worry too much if you can't get it."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "N, D = 100, 500\n", 
    "x = 5 * np.random.randn(N, D) + 12\n", 
    "gamma = np.random.randn(D)\n", 
    "beta = np.random.randn(D)\n", 
    "dout = np.random.randn(N, D)\n", 
    "\n", 
    "bn_param = {'mode': 'train'}\n", 
    "out, cache = batchnorm_forward(x, gamma, beta, bn_param)\n", 
    "\n", 
    "t1 = time.time()\n", 
    "dx1, dgamma1, dbeta1 = batchnorm_backward(dout, cache)\n", 
    "t2 = time.time()\n", 
    "dx2, dgamma2, dbeta2 = batchnorm_backward_alt(dout, cache)\n", 
    "t3 = time.time()\n", 
    "\n", 
    "print 'dx difference: ', rel_error(dx1, dx2)\n", 
    "print 'dgamma difference: ', rel_error(dgamma1, dgamma2)\n", 
    "print 'dbeta difference: ', rel_error(dbeta1, dbeta2)\n", 
    "print 'speedup: %.2fx' % ((t2 - t1) / (t3 - t2))"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "## Fully Connected Nets with Batch Normalization\n", 
    "Now that you have a working implementation for batch normalization, go back to your `FullyConnectedNet` in the file `cs2312n/classifiers/fc_net.py`. Modify your implementation to add batch normalization.\n", 
    "\n", 
    "Concretely, when the flag `use_batchnorm` is `True` in the constructor, you should insert a batch normalization layer before each ReLU nonlinearity. The outputs from the last layer of the network should not be normalized. Once you are done, run the following to gradient-check your implementation.\n", 
    "\n", 
    "HINT: You might find it useful to define an additional helper layer similar to those in the file `cs231n/layer_utils.py`. If you decide to do so, do it in the file `cs231n/classifiers/fc_net.py`."
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
    "for reg in [0, 3.14]:\n", 
    "  print 'Running check with reg = ', reg\n", 
    "  model = FullyConnectedNet([H1, H2], input_dim=D, num_classes=C,\n", 
    "                            reg=reg, weight_scale=5e-2, dtype=np.float64,\n", 
    "                            use_batchnorm=True)\n", 
    "\n", 
    "  loss, grads = model.loss(X, y)\n", 
    "  print 'Initial loss: ', loss\n", 
    "\n", 
    "  for name in sorted(grads):\n", 
    "    f = lambda _: model.loss(X, y)[0]\n", 
    "    grad_num = eval_numerical_gradient(f, model.params[name], verbose=False, h=1e-5)\n", 
    "    print '%s relative error: %.2e' % (name, rel_error(grad_num, grads[name]))\n", 
    "  if reg == 0: print"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Batchnorm for deep networks\n", 
    "Run the following to train a six-layer network on a subset of 1000 training examples both with and without batch normalization."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Try training a very deep net with batchnorm\n", 
    "hidden_dims = [100, 100, 100, 100, 100]\n", 
    "\n", 
    "num_train = 1000\n", 
    "small_data = {\n", 
    "  'X_train': data['X_train'][:num_train],\n", 
    "  'y_train': data['y_train'][:num_train],\n", 
    "  'X_val': data['X_val'],\n", 
    "  'y_val': data['y_val'],\n", 
    "}\n", 
    "\n", 
    "weight_scale = 2e-2\n", 
    "bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)\n", 
    "model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)\n", 
    "\n", 
    "bn_solver = Solver(bn_model, small_data,\n", 
    "                num_epochs=10, batch_size=50,\n", 
    "                update_rule='adam',\n", 
    "                optim_config={\n", 
    "                  'learning_rate': 1e-3,\n", 
    "                },\n", 
    "                verbose=True, print_every=200)\n", 
    "bn_solver.train()\n", 
    "\n", 
    "solver = Solver(model, small_data,\n", 
    "                num_epochs=10, batch_size=50,\n", 
    "                update_rule='adam',\n", 
    "                optim_config={\n", 
    "                  'learning_rate': 1e-3,\n", 
    "                },\n", 
    "                verbose=True, print_every=200)\n", 
    "solver.train()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "Run the following to visualize the results from two networks trained above. You should find that using batch normalization helps the network to converge much faster."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "plt.subplot(3, 1, 1)\n", 
    "plt.title('Training loss')\n", 
    "plt.xlabel('Iteration')\n", 
    "\n", 
    "plt.subplot(3, 1, 2)\n", 
    "plt.title('Training accuracy')\n", 
    "plt.xlabel('Epoch')\n", 
    "\n", 
    "plt.subplot(3, 1, 3)\n", 
    "plt.title('Validation accuracy')\n", 
    "plt.xlabel('Epoch')\n", 
    "\n", 
    "plt.subplot(3, 1, 1)\n", 
    "plt.plot(solver.loss_history, 'o', label='baseline')\n", 
    "plt.plot(bn_solver.loss_history, 'o', label='batchnorm')\n", 
    "\n", 
    "plt.subplot(3, 1, 2)\n", 
    "plt.plot(solver.train_acc_history, '-o', label='baseline')\n", 
    "plt.plot(bn_solver.train_acc_history, '-o', label='batchnorm')\n", 
    "\n", 
    "plt.subplot(3, 1, 3)\n", 
    "plt.plot(solver.val_acc_history, '-o', label='baseline')\n", 
    "plt.plot(bn_solver.val_acc_history, '-o', label='batchnorm')\n", 
    "  \n", 
    "for i in [1, 2, 3]:\n", 
    "  plt.subplot(3, 1, i)\n", 
    "  plt.legend(loc='upper center', ncol=4)\n", 
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
    "# Batch normalization and initialization\n", 
    "We will now run a small experiment to study the interaction of batch normalization and weight initialization.\n", 
    "\n", 
    "The first cell will train 8-layer networks both with and without batch normalization using different scales for weight initialization. The second layer will plot training accuracy, validation set accuracy, and training loss as a function of the weight initialization scale."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "execution_count": null, 
   "cell_type": "code", 
   "source": [
    "# Try training a very deep net with batchnorm\n", 
    "hidden_dims = [50, 50, 50, 50, 50, 50, 50]\n", 
    "\n", 
    "num_train = 1000\n", 
    "small_data = {\n", 
    "  'X_train': data['X_train'][:num_train],\n", 
    "  'y_train': data['y_train'][:num_train],\n", 
    "  'X_val': data['X_val'],\n", 
    "  'y_val': data['y_val'],\n", 
    "}\n", 
    "\n", 
    "bn_solvers = {}\n", 
    "solvers = {}\n", 
    "weight_scales = np.logspace(-4, 0, num=20)\n", 
    "for i, weight_scale in enumerate(weight_scales):\n", 
    "  print 'Running weight scale %d / %d' % (i + 1, len(weight_scales))\n", 
    "  bn_model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=True)\n", 
    "  model = FullyConnectedNet(hidden_dims, weight_scale=weight_scale, use_batchnorm=False)\n", 
    "\n", 
    "  bn_solver = Solver(bn_model, small_data,\n", 
    "                  num_epochs=10, batch_size=50,\n", 
    "                  update_rule='adam',\n", 
    "                  optim_config={\n", 
    "                    'learning_rate': 1e-3,\n", 
    "                  },\n", 
    "                  verbose=False, print_every=200)\n", 
    "  bn_solver.train()\n", 
    "  bn_solvers[weight_scale] = bn_solver\n", 
    "\n", 
    "  solver = Solver(model, small_data,\n", 
    "                  num_epochs=10, batch_size=50,\n", 
    "                  update_rule='adam',\n", 
    "                  optim_config={\n", 
    "                    'learning_rate': 1e-3,\n", 
    "                  },\n", 
    "                  verbose=False, print_every=200)\n", 
    "  solver.train()\n", 
    "  solvers[weight_scale] = solver"
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
    "# Plot results of weight scale experiment\n", 
    "best_train_accs, bn_best_train_accs = [], []\n", 
    "best_val_accs, bn_best_val_accs = [], []\n", 
    "final_train_loss, bn_final_train_loss = [], []\n", 
    "\n", 
    "for ws in weight_scales:\n", 
    "  best_train_accs.append(max(solvers[ws].train_acc_history))\n", 
    "  bn_best_train_accs.append(max(bn_solvers[ws].train_acc_history))\n", 
    "  \n", 
    "  best_val_accs.append(max(solvers[ws].val_acc_history))\n", 
    "  bn_best_val_accs.append(max(bn_solvers[ws].val_acc_history))\n", 
    "  \n", 
    "  final_train_loss.append(np.mean(solvers[ws].loss_history[-100:]))\n", 
    "  bn_final_train_loss.append(np.mean(bn_solvers[ws].loss_history[-100:]))\n", 
    "  \n", 
    "plt.subplot(3, 1, 1)\n", 
    "plt.title('Best val accuracy vs weight initialization scale')\n", 
    "plt.xlabel('Weight initialization scale')\n", 
    "plt.ylabel('Best val accuracy')\n", 
    "plt.semilogx(weight_scales, best_val_accs, '-o', label='baseline')\n", 
    "plt.semilogx(weight_scales, bn_best_val_accs, '-o', label='batchnorm')\n", 
    "plt.legend(ncol=2, loc='lower right')\n", 
    "\n", 
    "plt.subplot(3, 1, 2)\n", 
    "plt.title('Best train accuracy vs weight initialization scale')\n", 
    "plt.xlabel('Weight initialization scale')\n", 
    "plt.ylabel('Best training accuracy')\n", 
    "plt.semilogx(weight_scales, best_train_accs, '-o', label='baseline')\n", 
    "plt.semilogx(weight_scales, bn_best_train_accs, '-o', label='batchnorm')\n", 
    "plt.legend()\n", 
    "\n", 
    "plt.subplot(3, 1, 3)\n", 
    "plt.title('Final training loss vs weight initialization scale')\n", 
    "plt.xlabel('Weight initialization scale')\n", 
    "plt.ylabel('Final training loss')\n", 
    "plt.semilogx(weight_scales, final_train_loss, '-o', label='baseline')\n", 
    "plt.semilogx(weight_scales, bn_final_train_loss, '-o', label='batchnorm')\n", 
    "plt.legend()\n", 
    "\n", 
    "plt.gcf().set_size_inches(10, 15)\n", 
    "plt.show()"
   ], 
   "outputs": [], 
   "metadata": {
    "collapsed": false
   }
  }, 
  {
   "source": [
    "# Question:\n", 
    "Describe the results of this experiment, and try to give a reason why the experiment gave the results that it did."
   ], 
   "cell_type": "markdown", 
   "metadata": {}
  }, 
  {
   "source": [
    "# Answer:\n"
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