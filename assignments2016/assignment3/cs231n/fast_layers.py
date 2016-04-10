import numpy as np
try:
  from cs231n.im2col_cython import col2im_cython, im2col_cython
  from cs231n.im2col_cython import col2im_6d_cython
except ImportError:
  print 'run the following from the cs231n directory and try again:'
  print 'python setup.py build_ext --inplace'
  print 'You may also need to restart your iPython kernel'

from cs231n.im2col import *


def conv_forward_im2col(x, w, b, conv_param):
  """
  A fast implementation of the forward pass for a convolutional layer
  based on im2col and col2im.
  """
  N, C, H, W = x.shape
  num_filters, _, filter_height, filter_width = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']

  # Check dimensions
  assert (W + 2 * pad - filter_width) % stride == 0, 'width does not work'
  assert (H + 2 * pad - filter_height) % stride == 0, 'height does not work'

  # Create output
  out_height = (H + 2 * pad - filter_height) / stride + 1
  out_width = (W + 2 * pad - filter_width) / stride + 1
  out = np.zeros((N, num_filters, out_height, out_width), dtype=x.dtype)

  # x_cols = im2col_indices(x, w.shape[2], w.shape[3], pad, stride)
  x_cols = im2col_cython(x, w.shape[2], w.shape[3], pad, stride)
  res = w.reshape((w.shape[0], -1)).dot(x_cols) + b.reshape(-1, 1)

  out = res.reshape(w.shape[0], out.shape[2], out.shape[3], x.shape[0])
  out = out.transpose(3, 0, 1, 2)

  cache = (x, w, b, conv_param, x_cols)
  return out, cache


def conv_forward_strides(x, w, b, conv_param):
  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  stride, pad = conv_param['stride'], conv_param['pad']

  # Check dimensions
  #assert (W + 2 * pad - WW) % stride == 0, 'width does not work'
  #assert (H + 2 * pad - HH) % stride == 0, 'height does not work'

  # Pad the input
  p = pad
  x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
  
  # Figure out output dimensions
  H += 2 * pad
  W += 2 * pad
  out_h = (H - HH) / stride + 1
  out_w = (W - WW) / stride + 1

  # Perform an im2col operation by picking clever strides
  shape = (C, HH, WW, N, out_h, out_w)
  strides = (H * W, W, 1, C * H * W, stride * W, stride)
  strides = x.itemsize * np.array(strides)
  x_stride = np.lib.stride_tricks.as_strided(x_padded,
                shape=shape, strides=strides)
  x_cols = np.ascontiguousarray(x_stride)
  x_cols.shape = (C * HH * WW, N * out_h * out_w)

  # Now all our convolutions are a big matrix multiply
  res = w.reshape(F, -1).dot(x_cols) + b.reshape(-1, 1)

  # Reshape the output
  res.shape = (F, N, out_h, out_w)
  out = res.transpose(1, 0, 2, 3)

  # Be nice and return a contiguous array
  # The old version of conv_forward_fast doesn't do this, so for a fair
  # comparison we won't either
  out = np.ascontiguousarray(out)

  cache = (x, w, b, conv_param, x_cols)
  return out, cache
  

def conv_backward_strides(dout, cache):
  x, w, b, conv_param, x_cols = cache
  stride, pad = conv_param['stride'], conv_param['pad']

  N, C, H, W = x.shape
  F, _, HH, WW = w.shape
  _, _, out_h, out_w = dout.shape

  db = np.sum(dout, axis=(0, 2, 3))

  dout_reshaped = dout.transpose(1, 0, 2, 3).reshape(F, -1)
  dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

  dx_cols = w.reshape(F, -1).T.dot(dout_reshaped)
  dx_cols.shape = (C, HH, WW, N, out_h, out_w)
  dx = col2im_6d_cython(dx_cols, N, C, H, W, HH, WW, pad, stride)

  return dx, dw, db


def conv_backward_im2col(dout, cache):
  """
  A fast implementation of the backward pass for a convolutional layer
  based on im2col and col2im.
  """
  x, w, b, conv_param, x_cols = cache
  stride, pad = conv_param['stride'], conv_param['pad']

  db = np.sum(dout, axis=(0, 2, 3))

  num_filters, _, filter_height, filter_width = w.shape
  dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(num_filters, -1)
  dw = dout_reshaped.dot(x_cols.T).reshape(w.shape)

  dx_cols = w.reshape(num_filters, -1).T.dot(dout_reshaped)
  # dx = col2im_indices(dx_cols, x.shape, filter_height, filter_width, pad, stride)
  dx = col2im_cython(dx_cols, x.shape[0], x.shape[1], x.shape[2], x.shape[3],
                     filter_height, filter_width, pad, stride)

  return dx, dw, db


conv_forward_fast = conv_forward_strides
conv_backward_fast = conv_backward_strides


def max_pool_forward_fast(x, pool_param):
  """
  A fast implementation of the forward pass for a max pooling layer.

  This chooses between the reshape method and the im2col method. If the pooling
  regions are square and tile the input image, then we can use the reshape
  method which is very fast. Otherwise we fall back on the im2col method, which
  is not much faster than the naive method.
  """
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  same_size = pool_height == pool_width == stride
  tiles = H % pool_height == 0 and W % pool_width == 0
  if same_size and tiles:
    out, reshape_cache = max_pool_forward_reshape(x, pool_param)
    cache = ('reshape', reshape_cache)
  else:
    out, im2col_cache = max_pool_forward_im2col(x, pool_param)
    cache = ('im2col', im2col_cache)
  return out, cache


def max_pool_backward_fast(dout, cache):
  """
  A fast implementation of the backward pass for a max pooling layer.

  This switches between the reshape method an the im2col method depending on
  which method was used to generate the cache.
  """
  method, real_cache = cache
  if method == 'reshape':
    return max_pool_backward_reshape(dout, real_cache)
  elif method == 'im2col':
    return max_pool_backward_im2col(dout, real_cache)
  else:
    raise ValueError('Unrecognized method "%s"' % method)


def max_pool_forward_reshape(x, pool_param):
  """
  A fast implementation of the forward pass for the max pooling layer that uses
  some clever reshaping.

  This can only be used for square pooling regions that tile the input.
  """
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']
  assert pool_height == pool_width == stride, 'Invalid pool params'
  assert H % pool_height == 0
  assert W % pool_height == 0
  x_reshaped = x.reshape(N, C, H / pool_height, pool_height,
                         W / pool_width, pool_width)
  out = x_reshaped.max(axis=3).max(axis=4)

  cache = (x, x_reshaped, out)
  return out, cache


def max_pool_backward_reshape(dout, cache):
  """
  A fast implementation of the backward pass for the max pooling layer that
  uses some clever broadcasting and reshaping.

  This can only be used if the forward pass was computed using
  max_pool_forward_reshape.

  NOTE: If there are multiple argmaxes, this method will assign gradient to
  ALL argmax elements of the input rather than picking one. In this case the
  gradient will actually be incorrect. However this is unlikely to occur in
  practice, so it shouldn't matter much. One possible solution is to split the
  upstream gradient equally among all argmax elements; this should result in a
  valid subgradient. You can make this happen by uncommenting the line below;
  however this results in a significant performance penalty (about 40% slower)
  and is unlikely to matter in practice so we don't do it.
  """
  x, x_reshaped, out = cache

  dx_reshaped = np.zeros_like(x_reshaped)
  out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
  mask = (x_reshaped == out_newaxis)
  dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
  dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
  dx_reshaped[mask] = dout_broadcast[mask]
  dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
  dx = dx_reshaped.reshape(x.shape)

  return dx


def max_pool_forward_im2col(x, pool_param):
  """
  An implementation of the forward pass for max pooling based on im2col.

  This isn't much faster than the naive version, so it should be avoided if
  possible.
  """
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  assert (H - pool_height) % stride == 0, 'Invalid height'
  assert (W - pool_width) % stride == 0, 'Invalid width'

  out_height = (H - pool_height) / stride + 1
  out_width = (W - pool_width) / stride + 1

  x_split = x.reshape(N * C, 1, H, W)
  x_cols = im2col(x_split, pool_height, pool_width, padding=0, stride=stride)
  x_cols_argmax = np.argmax(x_cols, axis=0)
  x_cols_max = x_cols[x_cols_argmax, np.arange(x_cols.shape[1])]
  out = x_cols_max.reshape(out_height, out_width, N, C).transpose(2, 3, 0, 1)

  cache = (x, x_cols, x_cols_argmax, pool_param)
  return out, cache


def max_pool_backward_im2col(dout, cache):
  """
  An implementation of the backward pass for max pooling based on im2col.

  This isn't much faster than the naive version, so it should be avoided if
  possible.
  """
  x, x_cols, x_cols_argmax, pool_param = cache
  N, C, H, W = x.shape
  pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
  stride = pool_param['stride']

  dout_reshaped = dout.transpose(2, 3, 0, 1).flatten()
  dx_cols = np.zeros_like(x_cols)
  dx_cols[x_cols_argmax, np.arange(dx_cols.shape[1])] = dout_reshaped
  dx = col2im_indices(dx_cols, (N * C, 1, H, W), pool_height, pool_width,
              padding=0, stride=stride)
  dx = dx.reshape(x.shape)

  return dx
