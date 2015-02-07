---
layout: page
permalink: /convolutional-networks/
---

(These notes are currently in draft form and under development)

Table of Contents

## Convolutional Neural Networks (CNNs / ConvNets)

Convolutional Neural Networks are very similar to ordinary Neural Networks: They are made up of neurons that receive some inputs, perform a dot product with their weight vector and follow the dot product with a non-linearity. The whole network still express a single differentiable score function, from the image on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last fully-connected layer and all the tips/tricks we developed for learning regular Neural Networks still apply. 

So what does change? ConvNet architectures make the explicit assumption that the inputs are images, which allows us to be more efficient in how we arrange the neurons in the Neural Network and how we connect them together.

### Architecture Overview

*Recall: Regular Neural Nets.* As we saw in the previous chapter, Neural Networks receive an input (a single vector), and transform it through a series of *hidden layers*. Each hidden layer is made up of a set of neurons, where each neuron is fully connected to all neurons in the previous layer, and where neurons in a single layer function completely independently and do not share any connections. The last fully-connected layer is called the "output layer" and in classification settings it represents the class scores.

*Regular Neural Nets don't scale well to full images*. In CIFAR-10, images are only of size 32x32x3 (32 wide, 32 high, 3 color channels), so a single neuron in a first hidden layer of a regular Neural Network will have 32\*32\*3 = 3072 weights. That still seems manageable, but clearly this fully-connected structure does not scale to larger images. For example, an image of more respectible size, e.g. 200x200x3, would lead to neurons that have 200\*200\*3 = 120,000 weights. Moreover, we would almost certainly want to have several such neurons, so the parameters would add up quickly! Clearly, this full connectivity is wasteful and the huge number of parameters would quickly lead to overfitting.

*3D volumes of neurons*. Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way. In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: **width, height, depth**. For example, the input images in CIFAR-10 are an input volume of activations, and the volume has dimensions 32x32x3 (width, height, depth respectively). As we will soon see, the neurons in a layer will only be connected to a small region of the layer before it, instead of all of the neurons in a fully-connected manner. Moreover, the final output layer would for CIFAR-10 have dimensions 1x1x10, because by the end of the ConvNet architecture we will reduce the full image into a single vector of class scores, arranged along the depth. Here is a visualization:

<div class="fig figcenter fighighlight">
  <img src="/assets/nn1/neural_net2.jpeg" width="40%">
  <img src="/assets/cnn/cnn.jpeg" width="48%" style="border-left: 1px solid black;">
  <div class="figcaption">Left: A regular 3-layer Neural Network. Right: A ConvNet arranges its neurons in three dimensions (width, height, depth), as visualized in one of the layers. Every layer of a ConvNet transforms the 3D input volume to a 3D output volume of neuron activations. In this example, the red input layer holds the image, so its width and height would be the dimensions of the image, and the depth would be 3 (Red, Green, Blue channels).</div>
</div>

### Layers used to build ConvNets 

As we described above, every layer of a ConvNet transforms one volume of activations to another through a differentiable function. We use three main types of layers to build ConvNet architectures: Convolutional Layer, Pooling Layer, and Fully-Connected Layer (as seen in regular Neural Networks). We will eventually stack these layers to form a full ConvNet architecture. We now discuss these layers in more detail.

#### Convolutional Layer

The Conv layer is the core building block of a Convolutional Network. It is made up of neurons arranged in 3D volume. We now discuss the details of the neuron connectivities, their arrangement in space, and their parameter sharing scheme.

**Local Connectivity.** Instead of connecting to all neurons in the previous layer, each neuron connects to a local region of the input volume. The connections for each neuron are local in space (along width and height), but always full along the entire depth.

*Example*. For example, if an input volume has size [32x32x3], (e.g. an RGB CIFAR-10 image), then an example neuron in the convolutional layer might have a weight vector of size [5x5x3], for a total of 5\*5\*3 = 75 weights to the image. The spatial size 5x5 is a hyperparameter, and is also sometimes referred to as the size of the **receptive field** of the neuron. Notice that the the extent of the weights along the depth axis must be 3, since this is the depth of the input volume.

*Example 2*. Suppose an input volume had size [16x16x20]. Then using an example receptive field size of 3x3, every neuron in the Conv Layer would now have a total of 3\*3\*20 = 180 connections to the input volume. Notice that, again, the connectivity is local in space (e.g. 3x3), but full along depth (20).

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/depthcol.jpeg" width="40%">
  <img src="/assets/nn1/neuron_model.jpeg" width="40%" style="border-left: 1px solid black;">
  <div class="figcaption">
    <b>Left:</b> An example input volume in red (e.g. a 32x32x3 CIFAR-10 image), and an example volume of neurons in the first Convolutional layer. Each neuron in the convolutional layer is connected only to a local region in the input volume spatially, but to the full depth (i.e. all color channels). Note, there are multiple neurons (5 in this example) along the depth, all looking at the same region in the input - see discussion of depth columns in text below. <b>Right:</b> The neurons from the Neural Network chapter remain unchanged: They still compute a dot product of their weights with the input followed by a non-linearity, but their connectivity is now restricted to be local spatially.
  </div>
</div>

**Spatial arrangement**. We have explained the connectivity of each neuron in the Conv Layer to the input volume, but we haven't yet discussed how many neurons there are in the output volume or how they are arranged. Three hyperparameters control the size of the output volume:

1. First, the **depth** of the output volume is a hyperparameter; It controls the number of neurons in the Conv layer that all connect to the same region of the input volume. This is analogous to a regular Neural Network, where we had multiple neurons in a hidden layer all looking at the exact same input. However, all of these neurons activate for different features in the input. For example, if the first Convolutional Layer takes as input the raw image, then different neurons may activate in presence of various oriented edged, or blobs of color. We will refer to a set of neurons that are all looking at the same region of the input as a **depth column**.
2. Second, we must specify the **stride** with which we allocate depth columns around the spatial dimensions (width and height). When the stride is 1, then we will allocate a new depth column of neurons to spatial positions only 1 unit apart. This will lead to heavily overlapping receptive fields between the columns, and also to large output volumes. Conversely, if we use higher strides then the receptive fields will overlap less and the resulting output volume will have smaller dimensions spatially.
3. As we will soon see, sometimes it can be convenient to pad the input with zeros on the border spatially. The last hyperparameter that controls the output volume size is the size of this **zero-padding**. The nice feature of zero padding is that it will allow us to exactly preserve the spatial size when going from input volume to output volume. If we were to not use zero padding, then every Convolutional layer would reduce the spatial size of the volume, too quickly washing away information at the edges of the volumes.

We can compute the spatial size of the output volume as a function of the input volume size (\\(W\\)), the receptive field size of the neurons (\\(F\\)), the stride with which they are applied (\\(S\\)), and the amount of zero padding used (\\(P\\)) on the border. You can convince yourself that the correct formula for calculating how many neurons "fit" is given by \\((W - F + 2P)/S + 1\\). If this number is not an integer, then the strides are set incorrectly and the neurons cannot be tiled so that they "fit" across the volume neatly, in a symmetric way.

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/stride.jpeg">
  <div class="figcaption">
    Illustration of stride. In this example there is only one spatial dimension (x-axis), one neuron with a receptive field size of F = 3, the input size is W = 5, there is zero padding of P = 1. <b>Left:</b> The neuron strided across the input in stride of S = 1, giving output of size (5 - 3 + 2)/1+1 = 5. <b>Right:</b> The neuron uses stride of S = 2, giving output of size (5 - 3 + 2)/2+1 = 3. The neuron weights are in this example [1,0,-1] (shown on very right), and its bias is zero. These weights are shared across all yellow neurons (see parameter sharing below).
  </div>
</div>


**Parameter Sharing.** We can dramatically reduce the number of parameters used in a Conv Layer by assuming that every neuron in a single depth slice should use the same weights. We use the term **depth slice** as synonymous to an activation map; That is, it is the slice of the activation volume at a single depth, but for all spatial locations. The parameter sharing is motivated by the fact that if some particular feature (e.g. an oriented edge in a first convolutional layer) is useful at some position (x,y), then it should also be useful at some other position (x2,y2). Hence, there is no need to re-learn these features from scratch at all different locations in the output volume. In practice during backpropagation, every neuron in the volume will compute the gradient for its weights, but these gradients will be added up across each depth slice and only update a single set of weights. 

Additionally, if all neurons in a single depth slice are using the same weight vector, then the forward pass of the conv layer can in each depth slice be computed as a convolution of the neuron's weights (i.e. the filter), with the input volume. The result of this convolution is an *activation map*, and the set of activation maps for each different filter are stacked together along the depth dimension to produce the output volume.

*Example*. Suppose the input volume is a numpy array `X`. Then a single depth column at position `(x,y)` would be the activations `X[x,y,:]`. A depth slice at depth `d`, or equivalently an activation map at depth `d` would be the activations `X[:,:,d]`. 

*Example 2*. Suppose that the input volume `X` has shape `X.shape: (11,11,4)`. Suppose further that we use no zero padding (\\(P = 0\\)), that the filter size is \\(F = 5\\), and that the stride is \\(S = 2\\). The output volume would therefore have spatial size (11-5)/2+1 = 4, giving a volume with width and height of 4. The activation map in the output volume (call it `V`), would then look as follows (only some of the elements are computed in this example):

- `V[0,0,0] = np.sum(X[:5,:5,:] * W0) + b0`
- `V[1,0,0] = np.sum(X[2:7,:5,:] * W0) + b0`
- `V[2,0,0] = np.sum(X[4:9,:5,:] * W0) + b0`
- `V[3,0,0] = np.sum(X[6:11,:5,:] * W0) + b0`

notice that the weight vector `W0` is the weight vector of that neuron and `b0` is the bias. Here, `W0` is assumed to be of shape `W0.shape: (5,5,4)`, since the filter size is 5 and the depth of the input volume is 4. Notice that at each point, we are computing the dot product as seen before in ordinary neural networks. Notice that we are stepping along the input volume `X` spatially in steps of 2 (i.e. the stride). Also, we see that we are using the same weight and bias (due to parameter sharing), and where the dimensions along the width are increasing in steps of 2 (i.e. the stride). To construct a second activation map in the output volume, we would have:

- `V[0,0,1] = np.sum(X[:5,:5,:] * W1) + b1`
- `V[1,0,1] = np.sum(X[2:7,:5,:] * W1) + b1`
- `V[2,0,1] = np.sum(X[4:9,:5,:] * W1) + b1`
- `V[3,0,1] = np.sum(X[6:11,:5,:] * W1) + b1`

where we see that we are indexing into the second depth dimension in `V` because we are computing the second activation map, and that a different set of parameters (`W1`) is now used. Of course, this example only does the convolution in the x-axis, but the convolution works analogously in the y dimension. Also, remember that the activation maps should eventually be passed elementwise through an activation function such as ReLU, and this is not shown here.

**Summary**. To summarize, the Conv Layer specification is as follows:

- It accepts a volume of size \\(W\_1 \times H\_1 \times D\_1\\)
- It requires four hyperparameters: 
  - Number of filters \\(K\\), 
  - their spatial extent \\(F\\), 
  - the stride \\(S\\), 
  - the amount of zero padding \\(P\\).
- It produces a volume of size \\(W\_2 \times H\_2 \times D\_2\\) where:
  - \\(W\_2 = (W\_1 - F + 2P)/S + 1\\)
  - \\(H\_2 = (H\_1 - F + 2P)/S + 1\\) (i.e. width and height are computed equally by symmetry)
  - \\(D\_2 = K\\)
- With parameter sharing, it introduces \\(F \cdot F \cdot D\_1\\) weights per filter, for a total of \\((F \cdot F \cdot D\_1) \cdot K\\) weights and \\(K\\) biases.

#### Pooling Layer

The Pooling Layer operates independently on every depth slice of the input and resizes it spatially, using the MAX operation. For example, a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. Every MAX operation would in this case be taking a max over 4 numbers (little 2x2 region in some depth slice). The depth dimension remains unchanged. More specifically:

- It accepts a volume of size \\(W\_1 \times H\_1 \times D\_1\\)
- It requires three hyperparameters: 
  - their spatial extent \\(F\\), 
  - the stride \\(S\\), 
  - the amount of zero padding \\(P\\).
- It produces a volume of size \\(W\_2 \times H\_2 \times D\_2\\) where:
  - \\(W\_2 = (W\_1 - F + 2P)/S + 1\\)
  - \\(H\_2 = (H\_1 - F + 2P)/S + 1\\) (i.e. width and height are computed equally by symmetry)
  - \\(D\_2 = D\_1\\)
- It introduces zero parameters since it computes a fixed function of the input


#### Normalization Layer

Historically, various types of normalization layers were used but more recent work indicates that their contribution is minimal.

#### Fully-connected layer

Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks.

### ConvNet Architectures

- Common ways of arranging Conv Pool sandwiches
- Usually there are fewer filters near bottom of a Conv tower since there is less to know
- Case studies: LeNet, AlexNet, ZF net, VGG Net
- The product of filters and number of positions kept roughly constant
- On top: Fully Connected / Averaging

### Sizing Convolutional Networks

Case study of VGG network

- Most memory and compute is in first convolutional layers
- Most parameters are in the fully-connected layers at the end of the network

### Computational Considerations

- Add up the sizes of all volumes, multiply by 4 to get the number of bytes. Multiply by two because we need storage for forward pass (activations) and backward pass (gradients). 
- Add the number of bytes to store parameters
- Dividy number of bytes by 1024 to get number of KB, by 1024 to get number of MB, and 1024 again to get GB.
- Most GPUs currently have about 4GB of memory, or 6GB or up to 12GB. Use minibatch size that maxes out the memory in your GPU. Remember that smaller batch sizes likely need smaller learning rates.

### Misc Tips/Tricks

- Usually people apply less dropout in early conv layers since there are not that many parameters there compared to later stages of the network (e.g. the fully connected layers)
