import numpy as np
import h5py

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class PretrainedCNN(object):
  def __init__(self, dtype=np.float32, num_classes=100, input_size=64, h5_file=None):
    self.dtype = dtype
    self.conv_params = []
    self.input_size = input_size
    self.num_classes = num_classes
    
    # TODO: In the future it would be nice if the architecture could be loaded from
    # the HDF5 file rather than being hardcoded. For now this will have to do.
    self.conv_params.append({'stride': 2, 'pad': 2})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 2, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 2, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 2, 'pad': 1})
    self.conv_params.append({'stride': 1, 'pad': 1})
    self.conv_params.append({'stride': 2, 'pad': 1})

    self.filter_sizes = [5, 3, 3, 3, 3, 3, 3, 3, 3]
    self.num_filters = [64, 64, 128, 128, 256, 256, 512, 512, 1024]
    hidden_dim = 512

    self.bn_params = []
    
    cur_size = input_size
    prev_dim = 3
    self.params = {}
    for i, (f, next_dim) in enumerate(zip(self.filter_sizes, self.num_filters)):
      fan_in = f * f * prev_dim
      self.params['W%d' % (i + 1)] = np.sqrt(2.0 / fan_in) * np.random.randn(next_dim, prev_dim, f, f)
      self.params['b%d' % (i + 1)] = np.zeros(next_dim)
      self.params['gamma%d' % (i + 1)] = np.ones(next_dim)
      self.params['beta%d' % (i + 1)] = np.zeros(next_dim)
      self.bn_params.append({'mode': 'train'})
      prev_dim = next_dim
      if self.conv_params[i]['stride'] == 2: cur_size /= 2
    
    # Add a fully-connected layers
    fan_in = cur_size * cur_size * self.num_filters[-1]
    self.params['W%d' % (i + 2)] = np.sqrt(2.0 / fan_in) * np.random.randn(fan_in, hidden_dim)
    self.params['b%d' % (i + 2)] = np.zeros(hidden_dim)
    self.params['gamma%d' % (i + 2)] = np.ones(hidden_dim)
    self.params['beta%d' % (i + 2)] = np.zeros(hidden_dim)
    self.bn_params.append({'mode': 'train'})
    self.params['W%d' % (i + 3)] = np.sqrt(2.0 / hidden_dim) * np.random.randn(hidden_dim, num_classes)
    self.params['b%d' % (i + 3)] = np.zeros(num_classes)
    
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

    if h5_file is not None:
      self.load_weights(h5_file)

  
  def load_weights(self, h5_file, verbose=False):
    """
    Load pretrained weights from an HDF5 file.

    Inputs:
    - h5_file: Path to the HDF5 file where pretrained weights are stored.
    - verbose: Whether to print debugging info
    """

    # Before loading weights we need to make a dummy forward pass to initialize
    # the running averages in the bn_pararams
    x = np.random.randn(1, 3, self.input_size, self.input_size)
    y = np.random.randint(self.num_classes, size=1)
    loss, grads = self.loss(x, y)

    with h5py.File(h5_file, 'r') as f:
      for k, v in f.iteritems():
        v = np.asarray(v)
        if k in self.params:
          if verbose: print k, v.shape, self.params[k].shape
          if v.shape == self.params[k].shape:
            self.params[k] = v.copy()
          elif v.T.shape == self.params[k].shape:
            self.params[k] = v.T.copy()
          else:
            raise ValueError('shapes for %s do not match' % k)
        if k.startswith('running_mean'):
          i = int(k[12:]) - 1
          assert self.bn_params[i]['running_mean'].shape == v.shape
          self.bn_params[i]['running_mean'] = v.copy()
          if verbose: print k, v.shape
        if k.startswith('running_var'):
          i = int(k[11:]) - 1
          assert v.shape == self.bn_params[i]['running_var'].shape
          self.bn_params[i]['running_var'] = v.copy()
          if verbose: print k, v.shape
        
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(self.dtype)

  
  def forward(self, X, start=None, end=None, mode='test'):
    """
    Run part of the model forward, starting and ending at an arbitrary layer,
    in either training mode or testing mode.

    You can pass arbitrary input to the starting layer, and you will receive
    output from the ending layer and a cache object that can be used to run
    the model backward over the same set of layers.

    For the purposes of this function, a "layer" is one of the following blocks:

    [conv - spatial batchnorm - relu] (There are 9 of these)
    [affine - batchnorm - relu] (There is one of these)
    [affine] (There is one of these)

    Inputs:
    - X: The input to the starting layer. If start=0, then this should be an
      array of shape (N, C, 64, 64).
    - start: The index of the layer to start from. start=0 starts from the first
      convolutional layer. Default is 0.
    - end: The index of the layer to end at. start=11 ends at the last
      fully-connected layer, returning class scores. Default is 11.
    - mode: The mode to use, either 'test' or 'train'. We need this because
      batch normalization behaves differently at training time and test time.

    Returns:
    - out: Output from the end layer.
    - cache: A cache object that can be passed to the backward method to run the
      network backward over the same range of layers.
    """
    X = X.astype(self.dtype)
    if start is None: start = 0
    if end is None: end = len(self.conv_params) + 1
    layer_caches = []

    prev_a = X
    for i in xrange(start, end + 1):
      i1 = i + 1
      if 0 <= i < len(self.conv_params):
        # This is a conv layer
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]
        gamma, beta = self.params['gamma%d' % i1], self.params['beta%d' % i1]
        conv_param = self.conv_params[i]
        bn_param = self.bn_params[i]
        bn_param['mode'] = mode

        next_a, cache = conv_bn_relu_forward(prev_a, w, b, gamma, beta, conv_param, bn_param)
      elif i == len(self.conv_params):
        # This is the fully-connected hidden layer
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]
        gamma, beta = self.params['gamma%d' % i1], self.params['beta%d' % i1]
        bn_param = self.bn_params[i]
        bn_param['mode'] = mode
        next_a, cache = affine_bn_relu_forward(prev_a, w, b, gamma, beta, bn_param)
      elif i == len(self.conv_params) + 1:
        # This is the last fully-connected layer that produces scores
        w, b = self.params['W%d' % i1], self.params['b%d' % i1]
        next_a, cache = affine_forward(prev_a, w, b)
      else:
        raise ValueError('Invalid layer index %d' % i)

      layer_caches.append(cache)
      prev_a = next_a

    out = prev_a
    cache = (start, end, layer_caches)
    return out, cache


  def backward(self, dout, cache):
    """
    Run the model backward over a sequence of layers that were previously run
    forward using the self.forward method.

    Inputs:
    - dout: Gradient with respect to the ending layer; this should have the same
      shape as the out variable returned from the corresponding call to forward.
    - cache: A cache object returned from self.forward.

    Returns:
    - dX: Gradient with respect to the start layer. This will have the same
      shape as the input X passed to self.forward.
    - grads: Gradient of all parameters in the layers. For example if you run
      forward through two convolutional layers, then on the corresponding call
      to backward grads will contain the gradients with respect to the weights,
      biases, and spatial batchnorm parameters of those two convolutional
      layers. The grads dictionary will therefore contain a subset of the keys
      of self.params, and grads[k] and self.params[k] will have the same shape.
    """
    start, end, layer_caches = cache
    dnext_a = dout
    grads = {}
    for i in reversed(range(start, end + 1)):
      i1 = i + 1
      if i == len(self.conv_params) + 1:
        # This is the last fully-connected layer
        dprev_a, dw, db = affine_backward(dnext_a, layer_caches.pop())
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
      elif i == len(self.conv_params):
        # This is the fully-connected hidden layer
        temp = affine_bn_relu_backward(dnext_a, layer_caches.pop())
        dprev_a, dw, db, dgamma, dbeta = temp
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
        grads['gamma%d' % i1] = dgamma
        grads['beta%d' % i1] = dbeta
      elif 0 <= i < len(self.conv_params):
        # This is a conv layer
        temp = conv_bn_relu_backward(dnext_a, layer_caches.pop())
        dprev_a, dw, db, dgamma, dbeta = temp
        grads['W%d' % i1] = dw
        grads['b%d' % i1] = db
        grads['gamma%d' % i1] = dgamma
        grads['beta%d' % i1] = dbeta
      else:
        raise ValueError('Invalid layer index %d' % i)
      dnext_a = dprev_a

    dX = dnext_a
    return dX, grads


  def loss(self, X, y=None):
    """
    Classification loss used to train the network.

    Inputs:
    - X: Array of data, of shape (N, 3, 64, 64)
    - y: Array of labels, of shape (N,)

    If y is None, then run a test-time forward pass and return:
    - scores: Array of shape (N, 100) giving class scores.

    If y is not None, then run a training-time forward and backward pass and
    return a tuple of:
    - loss: Scalar giving loss
    - grads: Dictionary of gradients, with the same keys as self.params.
    """
    # Note that we implement this by just caling self.forward and self.backward
    mode = 'test' if y is None else 'train'
    scores, cache = self.forward(X, mode=mode)
    if mode == 'test':
      return scores
    loss, dscores = softmax_loss(scores, y)
    dX, grads = self.backward(dscores, cache)
    return loss, grads

