import os, json
import numpy as np
import h5py


def load_coco_data(base_dir='cs231n/datasets/coco_captioning',
                   max_train=None,
                   pca_features=True):
  data = {}
  caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
  with h5py.File(caption_file, 'r') as f:
    for k, v in f.iteritems():
      data[k] = np.asarray(v)

  if pca_features:
    train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7_pca.h5')
  else:
    train_feat_file = os.path.join(base_dir, 'train2014_vgg16_fc7.h5')
  with h5py.File(train_feat_file, 'r') as f:
    data['train_features'] = np.asarray(f['features'])

  if pca_features:
    val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7_pca.h5')
  else:
    val_feat_file = os.path.join(base_dir, 'val2014_vgg16_fc7.h5')
  with h5py.File(val_feat_file, 'r') as f:
    data['val_features'] = np.asarray(f['features'])

  dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
  with open(dict_file, 'r') as f:
    dict_data = json.load(f)
    for k, v in dict_data.iteritems():
      data[k] = v

  train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
  with open(train_url_file, 'r') as f:
    train_urls = np.asarray([line.strip() for line in f])
  data['train_urls'] = train_urls

  val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
  with open(val_url_file, 'r') as f:
    val_urls = np.asarray([line.strip() for line in f])
  data['val_urls'] = val_urls

  # Maybe subsample the training data
  if max_train is not None:
    num_train = data['train_captions'].shape[0]
    mask = np.random.randint(num_train, size=max_train)
    data['train_captions'] = data['train_captions'][mask]
    data['train_image_idxs'] = data['train_image_idxs'][mask]

  return data


def decode_captions(captions, idx_to_word):
  singleton = False
  if captions.ndim == 1:
    singleton = True
    captions = captions[None]
  decoded = []
  N, T = captions.shape
  for i in xrange(N):
    words = []
    for t in xrange(T):
      word = idx_to_word[captions[i, t]]
      if word != '<NULL>':
        words.append(word)
      if word == '<END>':
        break
    decoded.append(' '.join(words))
  if singleton:
    decoded = decoded[0]
  return decoded


def sample_coco_minibatch(data, batch_size=100, split='train'):
  split_size = data['%s_captions' % split].shape[0]
  mask = np.random.choice(split_size, batch_size)
  captions = data['%s_captions' % split][mask]
  image_idxs = data['%s_image_idxs' % split][mask]
  image_features = data['%s_features' % split][image_idxs]
  urls = data['%s_urls' % split][image_idxs]
  return captions, image_features, urls

