---
layout: page
permalink: /rnn/
---


Table of Contents:

- [Intro to RNN](#intro)




<a name='intro'></a>

## Intro to RNN

In this lecture note, we're going to be talking about the Recurrent Neural Networks (RNNs). One
great thing about the RNNs is that they offer a lot of flexibility on how we wire up the neural
network architecture. Normally when we're working with neural networks (Figure 1), we are given a fixed sized
input vector (red), then we process it with some hidden layers (green), and then we produce a
fixed sized output vector (blue) as depicted in the leftmost model in Figure 1. Recurrent Neural
Networks allow us to operate over sequences of input, output, or both at the same time. For
example, in the case of image captioning, we are given a fixed sized image and then through an RNN
we produce a sequence of words that describe the content of that image (second model in Figure 1).
Or for example, in the case of sentiment classification in the NLP, we are given a sequence of words
of the sentence and then we are trying to classify whether the sentiment of that sentence is
positive or negative (third model in Figure 1). In the case of machine translation, we can have an
RNN that takes a sequence of words of a sentence in English, and then this RNN is asked to produce
a sequence of words of a sentence in French, for example (forth model in Figure 1). As a last case,
we can have a video classification RNN where we might imagine classifying every single frame of
video with some number of classes, and most importantly we don't want the prediction to be only a
function of the current timestep (current frame of the video), but also all the timesteps (frames)
that have come before it in the video (rightmost model in Figure 1).

<div class="fig figcenter fighighlight">
  <img src="/assets/rnn/types.png" width="100%">
  <div class="figcaption">Different (non-exhaustive) types of Recurrent Neural Network architectures. Red boxes are input vectors. Green boxes are hidden layers. Blue boxes are output vectors.</div>
</div>
