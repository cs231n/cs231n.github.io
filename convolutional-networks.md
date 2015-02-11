---
layout: page
permalink: /convolutional-networks/
---

(These notes are currently in draft form and under development)

Table of Contents:

- [Architecture Overview](#overview)
- [ConvNet Layers](#layers)
  - [Convolutional Layer](#conv)
  - [Pooling Layer](#pool)
  - [Normalization Layer](#norm)
  - [Fully-Connected Layer](#fc)
- [ConvNet Architectures](#architectures)
  - [Layer Patterns](#layerpat)
  - [Layer Sizing Patterns](#layersizepat)
  - [Case Studies](#case) (AlexNet / ZFNet / VGGNet)
  - [Computational Considerations](#comp)
- [Additional References](#add)

## Convolutional Neural Networks (CNNs / ConvNets)

Convolutional Neural Networks are very similar to ordinary Neural Networks: They are made up of neurons that receive some inputs, perform a dot product with their weight vector and follow the dot product with a non-linearity. The whole network still express a single differentiable score function, from the image on one end to class scores at the other. And they still have a loss function (e.g. SVM/Softmax) on the last fully-connected layer and all the tips/tricks we developed for learning regular Neural Networks still apply. 

So what does change? ConvNet architectures make the explicit assumption that the inputs are images, which allows us to be more efficient about how we arrange the neurons and how we connect them together.

<a name='overview'></a>
### Architecture Overview

*Recall: Regular Neural Nets.* As we saw in the previous chapter, Neural Networks receive an input (a single vector), and transform it through a series of *hidden layers*. Each hidden layer is made up of a set of neurons, where each neuron is fully connected to all neurons in the previous layer, and where neurons in a single layer function completely independently and do not share any connections. The last fully-connected layer is called the "output layer" and in classification settings it represents the class scores.

*Regular Neural Nets don't scale well to full images*. In CIFAR-10, images are only of size 32x32x3 (32 wide, 32 high, 3 color channels), so a single neuron in a first hidden layer of a regular Neural Network will have 32\*32\*3 = 3072 weights. That still seems manageable, but clearly this fully-connected structure does not scale to larger images. For example, an image of more respectible size, e.g. 200x200x3, would lead to neurons that have 200\*200\*3 = 120,000 weights. Moreover, we would almost certainly want to have several such neurons, so the parameters would add up quickly! Clearly, this full connectivity is wasteful and the huge number of parameters would quickly lead to overfitting.

*3D volumes of neurons*. Convolutional Neural Networks take advantage of the fact that the input consists of images and they constrain the architecture in a more sensible way. In particular, unlike a regular Neural Network, the layers of a ConvNet have neurons arranged in 3 dimensions: **width, height, depth**. For example, the input images in CIFAR-10 are an input volume of activations, and the volume has dimensions 32x32x3 (width, height, depth respectively). As we will soon see, the neurons in a layer will only be connected to a small region of the layer before it, instead of all of the neurons in a fully-connected manner. Moreover, the final output layer would for CIFAR-10 have dimensions 1x1x10, because by the end of the ConvNet architecture we will reduce the full image into a single vector of class scores, arranged along the depth dimension. Here is a visualization:

<div class="fig figcenter fighighlight">
  <img src="/assets/nn1/neural_net2.jpeg" width="40%">
  <img src="/assets/cnn/cnn.jpeg" width="48%" style="border-left: 1px solid black;">
  <div class="figcaption">Left: A regular 3-layer Neural Network. Right: A ConvNet arranges its neurons in three dimensions (width, height, depth), as visualized in one of the layers. Every layer of a ConvNet transforms the 3D input volume to a 3D output volume of neuron activations. In this example, the red input layer holds the image, so its width and height would be the dimensions of the image, and the depth would be 3 (Red, Green, Blue channels).</div>
</div>

<a name='layers'></a>
### Layers used to build ConvNets 

As we described above, every layer of a ConvNet transforms one volume of activations to another through a differentiable function. We use three main types of layers to build ConvNet architectures: **Convolutional Layer**, **Pooling Layer**, and **Fully-Connected Layer** (exactly as seen in regular Neural Networks). We will stack these layers to form a full ConvNet architecture. 

*Example: Overview*. We will go into more details below, but the simplest ConvNet for CIFAR-10 classification could have the architecture [INPUT - CONV - RELU - POOL - FC]. In more detail:

- INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
- CONV layer will hold the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and the region they are connected to in the input volume. This may result in shape such as [32x32x12].
- RELU layer will apply an elementwise activation function, such as the \\(max(0,x)\\) thresholding at zero. This will not change the size of the volume ([32x32x12]).
- POOL layer will perform a downsampling operation along the spatial dimensions, resulting in volume such as [16x16x12]
- FC layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, each neuron in this layer will be connected densely to all numbers in the previous volume.

In this way, ConvNets transform the original image layer by layer from the original pixel values to the final class scores. The transformation is a function of a set of parameters (the weights of the neurons along the way), which we will train with gradient descent so that the class scores that the ConvNet computes are consistent with the labels in the training set for each image. We now describe the individual layers and the details of their connectivity in detail.

<a name='conv'></a>
#### Convolutional Layer

The Conv layer is the core building block of a Convolutional Network, and it made up of neurons arranged in a 3D volume. We now discuss the details of the neuron connectivities, their arrangement in space, and their parameter sharing scheme.

**Local Connectivity.** When dealing with high-dimensional inputs such as images, it is impractical to connect neurons to all neurons in the previous volume. Instead, we will connect each neuron to only a local region of the input volume. The spatial extent of this connectivity is a hyperparameter called the **receptive field** of the neuron. The extent of the connectivity along the depth axis is always equal to the depth of the input volume. It is important to note this asymmetry in how we treat the spatial dimensions (width and height) and the depth dimension: The connections are local in space (along width and height), but always full along the entire depth of the input volume.

*Example 1*. For example, suppose that the input volume has size [32x32x3], (e.g. an RGB CIFAR-10 image). If the receptive field is of size 5x5, then each neuron in the Conv Layer will have weights to a [5x5x3] region in the input volume, for a total of 5\*5\*3 = 75 weights. Notice that the the extent of the connectivity along the depth axis must be 3, since this is the depth of the input volume.

*Example 2*. Suppose an input volume had size [16x16x20]. Then using an example receptive field size of 3x3, every neuron in the Conv Layer would now have a total of 3\*3\*20 = 180 connections to the input volume. Notice that, again, the connectivity is local in space (e.g. 3x3), but full along the input depth (20).

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/depthcol.jpeg" width="40%">
  <img src="/assets/nn1/neuron_model.jpeg" width="40%" style="border-left: 1px solid black;">
  <div class="figcaption">
    <b>Left:</b> An example input volume in red (e.g. a 32x32x3 CIFAR-10 image), and an example volume of neurons in the first Convolutional layer. Each neuron in the convolutional layer is connected only to a local region in the input volume spatially, but to the full depth (i.e. all color channels). Note, there are multiple neurons (5 in this example) along the depth, all looking at the same region in the input - see discussion of depth columns in text below. <b>Right:</b> The neurons from the Neural Network chapter remain unchanged: They still compute a dot product of their weights with the input followed by a non-linearity, but their connectivity is now restricted to be local spatially.
  </div>
</div>

**Spatial arrangement**. We have explained the connectivity of each neuron in the Conv Layer to the input volume, but we haven't yet discussed how many neurons there are in the output volume or how they are arranged. Three hyperparameters control the size of the output volume: the **depth, stride** and **zero-padding**. We discuss these next:

1. First, the **depth** of the output volume is a hyperparameter that we can pick; It controls the number of neurons in the Conv layer that connect to the same region of the input volume. This is analogous to a regular Neural Network, where we had multiple neurons in a hidden layer all looking at the exact same input. As we will see, all of these neurons will learn to activate for different features in the input. For example, if the first Convolutional Layer takes as input the raw image, then different neurons along the depth dimension may activate in presence of various oriented edged, or blobs of color. We will refer to a set of neurons that are all looking at the same region of the input as a **depth column**.
2. Second, we must specify the **stride** with which we allocate depth columns around the spatial dimensions (width and height). When the stride is 1, then we will allocate a new depth column of neurons to spatial positions only 1 spatial unit apart. This will lead to heavily overlapping receptive fields between the columns, and also to large output volumes. Conversely, if we use higher strides then the receptive fields will overlap less and the resulting output volume will have smaller dimensions spatially.
3. As we will soon see, sometimes it will be convenient to pad the input with zeros spatially on the border of the input volume. The size of this **zero-padding** is a hyperparameter. The nice feature of zero padding is that it will allow us to control the spatial size of the output volumes. In particular, we will sometimes want to exactly preserve the spatial size of the input volume.

We can compute the spatial size of the output volume as a function of the input volume size (\\(W\\)), the receptive field size of the Conv Layer neurons (\\(F\\)), the stride with which they are applied (\\(S\\)), and the amount of zero padding used (\\(P\\)) on the border. You can convince yourself that the correct formula for calculating how many neurons "fit" is given by \\((W - F + 2P)/S + 1\\). If this number is not an integer, then the strides are set incorrectly and the neurons cannot be tiled so that they "fit" across the volume neatly, in a symmetric way. An example might help to get intuitions for this formula:

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/stride.jpeg">
  <div class="figcaption">
    Illustration of spatial arrangement. In this example there is only one spatial dimension (x-axis), one neuron with a receptive field size of F = 3, the input size is W = 5, and there is zero padding of P = 1. <b>Left:</b> The neuron strided across the input in stride of S = 1, giving output of size (5 - 3 + 2)/1+1 = 5. <b>Right:</b> The neuron uses stride of S = 2, giving output of size (5 - 3 + 2)/2+1 = 3. Notice that stride S = 3 could not be used since it wouldn't fit neatly across the volume. In terms of the equation, this can be determined since (5 - 3 + 2) = 4 is not divisible by 3. 
    <br>The neuron weights are in this example [1,0,-1] (shown on very right), and its bias is zero. These weights are shared across all yellow neurons (see parameter sharing below).
  </div>
</div>

*Use of zero-padding*. In the example above on left, note that the input dimension was 5 and the output dimension was equal: also 5. This worked out so because our receptive fields were 3 and we used zero padding of 1. If there was no zero-padding used, then the output volume would have had spatial dimension of only 3, because that it is how many neurons would have "fit" across the original input. In general, setting zero padding to be \\(P = (F - 1)/2\\) when the stride \\(S = 1\\) ensures that the input volume and output volume will have the same size spatially. It is very common to use zero-padding in this way and we will discuss the full reasons when we talk more about ConvNet architectures.

*Constraints on strides*. Note that the spatial arrangement hyperparameters have mutual constraints. For example, when the input has size \\(W = 10\\), no zero-padding is used \\(P = 0\\), and the filter size is \\(F = 3\\), then it would be impossible to use stride \\(S = 2\\), since \\((W - F + 2P)/S + 1 = (10 - 3 + 0) / 2 + 1 = 4.5\\), i.e. not an integer, indicating that the neurons don't "fit" neatly and symmetrically across the input. Therefore, this setting of the hyperparameters is considered to be invalid, and a ConvNet library would likely throw an exception. As we will see in the ConvNet architectures section, sizing the ConvNets appropriately so that all the dimensions "work out" can be a real headache, which the use of zero-padding and some design guidelines will significantly alleviate.

*Real-world example*. The [Krizhevsky et al.](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) architecture that won the ImageNet challenge in 2012 accepted images of size [227x227x3]. On the first Convolutional Layer, it used neurons with receptive field size \\(F = 11\\), stride \\(S = 4\\) and no zero padding \\(P = 0\\). The Conv layer had a depth of \\(K = 96\\). Since (227 - 11)/4 + 1 = 55, we can see that the Conv layer output volume had size [55x55x96]. Each of the 55\*55\*96 neurons in this volume was connected to a region of size [11x11x3] in the input volume.

**Parameter Sharing.** Parameter sharing scheme is used in Convolutional Layers to control the number of parameters. Using the real-world example above, we see that there are 55\*55\*96 = 290,400 neurons in the first Conv Layer, and each has 11\*11\*3 = 363 weights and 1 bias. Together, this adds up to 290400 * 364 = 105,705,600 paramaters on the first layer of the ConvNet alone. Clearly, this number is very high.

It turns out that we can dramatically reduce the number of parameters by making one reasonable assumption: That if one patch feature is useful to compute at some spatial position (x,y), then it should also be useful to compute at a different position (x2,y2). In other words, denoting a single 2-dimensional slice of depth as a **depth slice** (e.g. a volume of size [55x55x96] has 96 depth slices, each of size [55x55]), we are going to constraint the neurons in each depth slice to use the same weights and bias. With this parameter sharing scheme, the first Conv Layer in our example would now have only 96 unique set of weights (one for each depth slice), for a total of 96\*11\*11\*3 = 34,848 unique weights, or 34,944 parameters (+96 biases). Alternatively, all 55*55 neurons in each depth slice will now be using the same parameters. In practice during backpropagation, every neuron in the volume will compute the gradient for its weights, but these gradients will be added up across each depth slice and only update a single set of weights per slice. 

Notice that if all neurons in a single depth slice are using the same weight vector, then the forward pass of the conv layer can in each depth slice be computed as a **convolution** of the neuron's weights with the input volume (Hence the name: Convolutional Layer). Therefore, it is common to refer to the sets of weights as a **filter** (or a **kernel**), which is convolved with the input. The result of this convolution is an *activation map* (e.g. of size [55x55]), and the set of activation maps for each different filter are stacked together along the depth dimension to produce the output volume (e.g. [55x55x96]).

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/weights.jpeg">
  <div class="figcaption">
    Example filters learned by Krizhevsky et al. Each of the 96 filters shown here is of size [11x11x3], and each one is shared by the 55*55 neurons in one depth slice. Notice that the parameter sharing assumption is relatively reasonable: If detecting a horizontal edge is important at some location in the image, it should intuitively be useful at some other location as well due to the translationally-invariant structure of images. There is therefore no need to relearn to detect a horizontal edge at every one of the 55*55 distinct locations in the Conv layer output volume.
  </div>
</div>

Note that sometimes the parameter sharing assumption may not make sense. This is especially the case when the input images to a ConvNet have some specific centered structure, where we should expect, for example, that completely different features should be learned on one side of the image than another. One practical example is when the input are faces that have been centered in the image. You might expect that different eye-specific or hair-specific features could (and should) be learned in different spatial locations. In that case it is common to relax the parameter sharing scheme, and instead simply call the layer a **Locally-Connected Layer**.

*Naming Convention Example*. Suppose the input volume is a numpy array `X`. Then a single depth column at position `(x,y)` would be the activations `X[x,y,:]`. A depth slice at depth `d`, or equivalently an activation map at depth `d` would be the activations `X[:,:,d]`. 

*Conv Layer Example 1*. Suppose that the input volume `X` has shape `X.shape: (11,11,4)`. Suppose further that we use no zero padding (\\(P = 0\\)), that the filter size is \\(F = 5\\), and that the stride is \\(S = 2\\). The output volume would therefore have spatial size (11-5)/2+1 = 4, giving a volume with width and height of 4. The activation map in the output volume (call it `V`), would then look as follows (only some of the elements are computed in this example):

- `V[0,0,0] = np.sum(X[:5,:5,:] * W0) + b0`
- `V[1,0,0] = np.sum(X[2:7,:5,:] * W0) + b0`
- `V[2,0,0] = np.sum(X[4:9,:5,:] * W0) + b0`
- `V[3,0,0] = np.sum(X[6:11,:5,:] * W0) + b0`

notice that the weight vector `W0` is the weight vector of that neuron and `b0` is the bias. Here, `W0` is assumed to be of shape `W0.shape: (5,5,4)`, since the filter size is 5 and the depth of the input volume is 4. Notice that at each point, we are computing the dot product as seen before in ordinary neural networks. Also, we see that we are using the same weight and bias (due to parameter sharing), and where the dimensions along the width are increasing in steps of 2 (i.e. the stride). To construct a second activation map in the output volume, we would have:

- `V[0,0,1] = np.sum(X[:5,:5,:] * W1) + b1`
- `V[1,0,1] = np.sum(X[2:7,:5,:] * W1) + b1`
- `V[2,0,1] = np.sum(X[4:9,:5,:] * W1) + b1`
- `V[3,0,1] = np.sum(X[6:11,:5,:] * W1) + b1`
- `V[0,1,1] = np.sum(X[:5,2:7,:] * W1) + b1` (example of going along y)
- `V[2,3,1] = np.sum(X[4:9,6:11,:] * W1) + b1` (or along both)

where we see that we are indexing into the second depth dimension in `V` because we are computing the second activation map, and that a different set of parameters (`W1`) is now used. In the example above, we are for brevity leaving out some of the other operatations the Conv Layer would perform to fill the other parts of the output array `V`. Also, remember that the activation maps should eventually be passed elementwise through an activation function such as ReLU, but this is not shown.

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
- In the output volume, the \\(d\\)-th depth slice (of size \\(W\_2 \times H\_2\\)) is the result of performing a valid convolution of the \\(d\\)-th filter over the input volume (and offset by \\(d\\)-th bias).

There are common conventions and rules of thumb for assigning these parameters. See the [ConvNet architectures](#architectures) section below.

<a name='pool'></a>
#### Pooling Layer

It is common to periodically insert a Pooling layer in-between successive Conv layers in a ConvNet architecture. Its function is to progressively reduce the spatial size of the representation to reduce the amount of parameters and computation in the network, and hence to also control overfitting. The Pooling Layer operates independently on every depth slice of the input and resizes it spatially, using the MAX operation. The most common form is a pooling layer with filters of size 2x2 applied with a stride of 2 downsamples every depth slice in the input by 2 along both width and height, discarding 75% of the activations. Every MAX operation would in this case be taking a max over 4 numbers (little 2x2 region in some depth slice). The depth dimension remains unchanged. More generally, the pooling layer:

- It accepts a volume of size \\(W\_1 \times H\_1 \times D\_1\\)
- It requires three hyperparameters: 
  - their spatial extent \\(F\\), 
  - the stride \\(S\\), 
- It produces a volume of size \\(W\_2 \times H\_2 \times D\_2\\) where:
  - \\(W\_2 = (W\_1 - F)/S + 1\\)
  - \\(H\_2 = (H\_1 - F)/S + 1\\)
  - \\(D\_2 = D\_1\\)
- It introduces zero parameters since it computes a fixed function of the input
- Note that it is not common to use zero-padding for Pooling layers

It is worth noting that there are only two commonly seen variations of the max pooling layer found in practice: A pooling layer with \\(F = 3, S = 2\\) (also called overlapping pooling), and more commonly \\(F = 2, S = 2\\). Pooling sizes with larger receptive fields are too destructive.

**General pooling**. In addition to max pooling, the pooling units can also perform other functions, such as *average pooling* or even *L2-norm pooling*. Average pooling was often used historically but has recently fallen out of favor compared to the max pooling operation, which has been shown to work better in practice.

<div class="fig figcenter fighighlight">
  <img src="/assets/cnn/pool.jpeg">
  <div class="figcaption">
    Pooling layer downsamples the volume spatially, independently in each depth slice of the input volume. The most common downsampling operation is max, giving rise to <b>max pooling</b>. In this example, the input volume of size [224x224x64] is max pooled with filter size 2, stride 2 into output volume of size [112x112x64]. Here, each max would be taken over 4 numbers (little 2x2 square). Notice that the volume depth is preserved. Other operations you may see is average pooling, or L2 pooling.
  </div>
</div>

<a name='norm'></a>
#### Normalization Layer

Many types of normalization layers were previously proposed, with the intentions of implementing inhibition schemes seen in the brain. However, these have fallen out of favor because their contribution has been shown to be minimal, if any. For various types of normalizations, see the discussion in Alex Krizhevsky's [cuda-convnet library API](http://code.google.com/p/cuda-convnet/wiki/LayerParams#Local_response_normalization_layer_(same_map)).

<a name='fc'></a>
#### Fully-connected layer

Neurons in a fully connected layer have full connections to all activations in the previous layer, as seen in regular Neural Networks. Their activations can hence be computed with matrix multiplication followed by a bias offset. See the *Neural Network* section of the notes for more information.

<a name='architectures'></a>
### ConvNet Architectures

We have seen that Convolutional Networks are commonly made up of only three layer types: CONV, POOL (we assume Max pool unless stated otherwise) and FC (short for fully-connected). We will also explicitly write the RELU activation function as a layer, which applies elementwise non-linearity. In this section we discuss how these are commonly stacked together to form entire ConvNets. 

<a name='layerpat'></a>
#### Layer Patterns
The most common form of a ConvNet architecture stacks a few CONV-RELU layers, follows them with POOL layers, and repeats this pattern until the image has been merged spatially to a small size. At some point, it is common to transition to fully-connected layers. The last fully-connected layer holds the output, such as the class scores. In other words, the most common ConvNet architecture follows the pattern:

`INPUT -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K -> FC`

where the `*` indicates repetition, and the `POOL?` indicates an optional pooling layer. Moreover, `N >= 0` (and usually `N <= 3`), `M >= 0`, `K >= 0` (and usually `K < 3`). For example, here are some common ConvNet architectures you may see that follow this pattern:

- `INPUT -> FC`, implements a linear classifier. Here `N = M = K = 0`.
- `INPUT -> CONV -> RELU -> FC`
- `INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC`. Here we see that there is a single CONV layer between every POOL layer.
- `INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC` Here we see two CONV layers stacked before every POOL layer. This is generally a good idea for larger and deeper networks, because multiple stacked CONV layers can develop more complex features of the input volume before the destructive pooling operation.

*Prefer a stack of small filter CONV to one large receptive field CONV layer*. Suppose that you stack three 3x3 CONV layers on top of each other (with non-linearities in between, of course). In this arrangement, each neuron on the first CONV layer has a 3x3 view of the input volume. A neuron on the second CONV layer has a 3x3 view of the first CONV layer, and hence by extension a 5x5 view of the input volume. Similarly, a neuron on the third CONV layer has a 3x3 view of the 2nd CONV layer, and hence a 7x7 view of the input volume. Suppose that instead of these three layers of 3x3 CONV, we only wanted to use a single CONV layer with 7x7 receptive fields. These neurons would have a receptive field size of the input volume that is identical in spatial extent (7x7), but with several disadvantages. First, the neurons would be computing a linear function over the input, while the three stacks of CONV layers contain non-linearities that make their features more expressive. Second, if we suppose that all the volumes have \\(C\\) channels, then it can be seen that the single 7x7 CONV layer would contain \\(C \times (7 \times 7 \times C) = 49 C^2\\) parameters, while the three 3x3 CONV layers would only contain \\(3 \times (C \times (3 \times 3 \times C)) = 27 C^2\\) parameters. Intuitively, stacking CONV layers with tiny filters as opposed to having one CONV layer with big filters allows us to express more powerful features of the input, and with fewer parameters. As a practical disadvantage, we might need more memory to hold all the intermediate CONV layer results if we plan to do backpropagation.

<a name='layersizepat'></a>
#### Layer Sizing Patterns

Until now we've omitted mentions of common hyperparameters used in each of the layers in a ConvNet. We will first state the common rules of thumb for sizing the architectures and then follow the rules with a discussion of the motation:

The **input layer** (that contains the image) should be divisible by 2 many times. Common numbers include 32 (e.g. CIFAR-10), 64, 96 (e.g. STL-10), or 224 (e.g. common ImageNet ConvNets), 384, and 512.

The **conv layers** should be using small filters (e.g. 3x3 or at most 5x5), using a stride of \\(S = 1\\), and crucially, padding the input volume with zeros in such way that the conv layer does not alter the spatial dimensions of the input. That is, when \\(F = 3\\), then using \\(P = 1\\) will retain the original size of the input. When \\(F = 5\\), \\(P = 2\\). For a general \\(F\\), it can be seen that \\(P = (F - 1) / 2\\) preserves the input size. If you must use bigger filter sizes (such as 7x7 or so), it is only common to see this on the very first conv layer that is looking at the input image.

The **pool layers** are in charge of downsampling the spatial dimensions of the input. The most common setting is to use max-pooling with 2x2 receptive fields (i.e. \\(F = 2\\)), and with a stride of 2 (i.e. \\(S = 2\\)). Note that this discards exactly 75% of the activations in an input volume (due to downsampling by 2 in both width and height). Another sligthly less common setting is to use 3x3 receptive fields with a stride of 2, but this makes. It is very uncommon to see receptive field sizes for max pooling that are larger than 3 because the pooling is then too lossy and agressive. This usually leads to worse performance.

*Reducing sizing headaches.* The scheme presented above is pleasing because all the CONV layers preserve the spatial size of their input, while the POOL layers alone are in charge of down-sampling the volumes spatially. In an alternative scheme where we use strides greater than 1 or don't zero-pad the input in CONV layers, we would have to very carefully keep track of the input volumes throughout the CNN architecture and make sure that all strides and filters "work out", and that the ConvNet architecture is nicely and symmetrically wired.

*Why use stride of 1 in CONV?* Smaller strides work better in practice. Additionally, as already mentioned stride 1 allows us to leave all spatial down-sampling to the POOL layers, with the CONV layers only transforming the input volume depth-wise.

*Why use padding?* In addition to the aforementioned benefit of keeping the spatial sizes constant after CONV, doing this actually improves performance. If the CONV layers were to not zero-pad the inputs and only perform valid convolutions, then the size of the volumes would reduce by a small amount after each CONV, and the information at the borders would be "washed away" too quickly.

*Compromising based on memory constraints.* In some cases (especially early in the ConvNet architectures), the amount of memory can build up very quickly with the rules of thumb presented above. For example, filtering a 224x224x3 image with three 3x3 CONV layers with 64 filters each and padding 1 would create three activation volumes of size [224x224x64]. This amounts to a total of about 10 million activations, or 72MB of memory (per image, for both activations and gradients). Since GPUs are often bottlenecked by memory, it may be necessary to compromise. In practice, people prefer to make the compromise at only the first CONV layer of the network. For example, one compromise might be to use a first CONV layer with filter sizes of 7x7 and stride of 2 (as seen in a ZF net). As another example, an AlexNet uses filer sizes of 11x11 and stride of 4.

<a name='case'></a>
#### Case studies

- AlexNet
- Zeiler & Fergus (ZF) Net
- VGGNet

<a name='comp'></a>
#### Computational Considerations

The largest bottleneck to be aware of when constructing ConvNet architectures is the memory bottleneck. Many modern GPUs have a limit of 3/4/6GB memory, with the best GPUs having about 12GB of memory. There are three major sources of memory to keep track of:

- From the intermediate volume sizes: These are the raw number of **activations** at every layer of the ConvNet, and also their gradients (of equal size). Usually, most of the activations are on the earlier layers of a ConvNet (i.e. first Conv Layers). These are kept around because they are needed for backpropagation, but a clever implementation that runs a ConvNet only at test time could in principle reduce this by a huge amount, by only storing the current activations at any layer and discarding the previous activations on layers below.
- From the parameter sizes: These are the numbers that hold the network **parameters**, their gradients during backpropagation, and commonly also a step cache if the optimization is using momentum, Adagrad, or RMSProp. Therefore, the memory to store the parameter vector alone must usually be multiplied by a factor of at least 3 or so.
- Every ConvNet implementation has to maintain **miscellaneous** memory, such as the image data batches, perhaps their augmented versions, etc.

Once you have a rough estimate of the total number of values (for activations, gradients, and misc), the number should be converted to size in GB. Take the number of values, multiply by 4 to get the raw number of bytes (since every floating point is 4 bytes, or maybe by 8 for double precision), and then divide by 1024 multiple times to get the amount of memory in KB, MB, and finally GB. If your network doesn't fit, a common heuristic to "make it fit" is to decrease the batch size, since most of the memory is usually consumed by the activations.

<a name='add'></a>
### Additional Resources

- [DeepLearning.net tutorial](http://deeplearning.net/tutorial/lenet.html)
