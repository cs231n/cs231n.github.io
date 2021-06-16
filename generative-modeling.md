---
title: 'Generative Modeling'
layout: page
permalink: /generative-modeling/
---


Table of Contents
- [Motivation and Overview](#Motivation-and-Overview)
- [Pixel RNN/CNN](#Pixel-RNN/CNN)
    - [Explicit density model](#Explicit-density-model)
    - [Pixel RNN](#Pixel-RNN)
    - [Pixel CNN](#Pixel-CNN)
- [Variational Autoencoder](#Variational-Autoencoder)
    - [Overview of Variational Autoencoder](#Overview-of-Variational-Autoencoder)
    - [Autoencoders v.s. Variational Autoencoders](#Autoencoders-v.s.-Variational-Autoencoders)
    - [VAE Mathematical Explanation](#VAE-Mathematical-Explanation)
    - [VAE training process](#VAE-training-process)
- [Generative Adversarial Networks](#Generative-Adversarial-Networks)
    - [Overview of Generative Adversarial Networks](#Overview-of-Generative-Adversarial-Networks) 
    - [Generative Adversarial Nets - 2014 original version](#Generative-Adversarial-Nets---2014-original-version)
        - [Discriminator network](#Discriminator-network)
        - [Generator network](#Generator-network)
    - [GAN Mathematical explanation](#GAN-Mathematical-explanation)
    - [Evaluation](#Evaluation)
        - [Inception Scores](#Inception-Scores)
        - [Nearest Neighbours](#Nearest-Neighbours)
        - [HYPE - Human eye perceptual Evaluation](#HYPE---Human-eye-perceptual-Evaluation)
    - [Challenges](#Challenges)
        - [Optimization](#Optimization)
        - [Mode Collapse](#Mode-collapse)
    - [Case Studies](#Case-Studies)
        - [DCGAN](#DCGAN)
        - [CycleGAN](#CycleGAN)
        - [StyleGAN](#StyleGAN)
    - [Summary](#Summary)

<a name='intro'></a>

## Motivation and Overview
In the first half of the quarter, we studied several supervised learning methods, which learn functions to map input images to labels. However, labeling the training data may be expensive because it requires much time and effort. Thus, we are introducing unsupervised learning methods. In unsupervised learning methods, training data is relatively cheaper because the methods don't need labeling from the huge dataset. The goal is to learn the underlying hidden structures or feature representations from raw data directly.

This table compares supervised and unsupervised learning: 
|          | Supervised Learning | Unsupervised Learning |
| -------- | -------- | -------- |
| Data         | has label y         |    no labels      |
| Goal         |  input data -> output label        |    Learn some underlying hidden strcture of the data      |
| Examples     | Classification, regression, object detection, semantic segmentation, image captioning, etc     |   Clustering, dimensionality reduction, feature learning, density estimation, etc.   |


**Generative modeling** is in the class of unsupervised learning. The goal of generative modeling is to generate new samples from the same distribution. In the application of image generation, we want to make sure that the quality of generated images aligns with the raw data image distributions. Thus, during the training process, there are two objectives:
1. Learn $p_{model} (x)$ that **approximates** $p_{data}(x)$ 
2. Sampling new data $x$ from $p_{model}(x)$ 


Within the first objective (how $p_{model} (x)$ approximates $p_{data}(x)$), we can categorize generative model into two types: 
1. Explicit density estimation
2. Implicit density estimation

![](https://i.imgur.com/L6l6qLn.png)



In this document, we will talk about 3 most popular types: 
1. Pixel RNN/CNN - Explicit density estimation
2. Variational Autoencoder - Approximate density
3. Generative Adversarial Networks - Implicit density

## Pixel RNN/CNN


### Explicit density model
**Pixel RNN/CNN** is an explicit density estimation method, which means that we explicitly define and solve for $p_{model}(x)$. For example, given an input image $x$, we can calculate the joint likelihood of each pixel in the image, in order to predict the of the pixel values of the image $x$. Being able to explicitly calculate the joint likelihood is why we called it explicitly defined method. 

However, estimating the joint likelihood of pixels directly can be difficult. A trick borrowed from probability is to rewrite the joint likelihood as a product of the conditional likelihoods on the previous pixels. This uses the chain rule to decompose the likelihood of an image $x$ into products of 1-dimensional densities. Our objective is then to maximize the likelihood of training data.


$$
\begin{aligned}
p(x) = p(x_1, x_2, ..., x_n)
     &= \prod_{i=1}^n p(x_i | x_1, ..., x_{i - 1}).
\end{aligned}
$$


### Pixel RNN
You may notice that the distribution of $p(x)$ is very complex because each pixel is conditioned on hundreds of thousands of pixels, making the computation very expensive. How can we resolve this problem? 

Recall RNN that we learn in the previous lecture. RNN has an “internal state” that is updated as a sequence is processed and allows previous outputs to be used as inputs. We can treat the conditional distribution of each pixel as a sequence of data, and apply RNN to model the joint likelihood function. More specifically, we can *model the dependency of one pixel on all the previous pixels by keeping the hidden state of all the previous inputs*. We use the hidden state to express the dependency of generating a new pixel from the previous pixel. In the beginning, we have the default hidden state,  first pixel $x_1$, and the image we want to generate. We will use the first pixel to generate the second pixel and repeat. Then it becomes a sequential generating process: feeding the previously predicted pixel back to the network to generate the next pixel.


The process described in the Pixel RNN paper is as follows: Starting from the corner, each pixel is conditional on the pixel from the left and the pixel above. Repeat the sequential generating process until the whole image is generated.

![](https://i.imgur.com/6ajn4Pe.gif)
<figcaption> Pixel RNN Sequential Generating Process </figcaption>


### Pixel CNN
One drawback of Pixel RNN is that the sequential generation is slow and we need to process the pixels one at a time. Is there a way to process more pixels at a time? In the same paper, the authors proposed another method, **Pixel CNN**, which allows parallelizaton among pixels. Specifically, Pixel CNN uses a *masked convolution over context region*. Different from regular square receptive field in the convolutional layer, the receptive field of masked convolution need not be a square. 

You may wonder: are we able to generate the whole image with masked convolution? In fact, if we stack enough layers of this kind of masked convolution, we can achieve the same effective receptive field as the pixel generation that conditional on all of the previous pixels (pixel RNN).

![](https://i.imgur.com/2eSJZ2Y.png)
<figcaption> Pixel CNN Generating Example </figcaption>



## Variational Autoencoder

### Overview of Variational Autoencoder

The modeling procedure of Pixel RNN is still slow because it's a sequential generation process. What if we make a little bit of trade-off but we can generate all pixels at the same time and model a simpler data distribution? Instead of optimizing the expensive tractable density function directly, we can *derive and optimize the lower bound on likelihood* instead. This is called Approximate density estimation.

We can re-write the probability density function as $p_{\theta}(x) = \int p_{\theta}(z)p_{\theta}(x|z)dz$. We introduce a new latent variable $z$ to *decompose the data likelihood as the marginal distribution of the conditional likelihood w.r.t this latent variable*. The latent variable $z$ represents *the underlying structure of the data distribution*. 

This method is called **Variational Autoencoders** (VAE). There is no dependency among the pixels. All pixels are conditional on this variable z. We can generate all pixels at the same time. The drawback is we need to integrate all possible values of z. In reality, $p_\theta(z)$ is low dimensional and $p_\theta(x|z)$ is often times complex, making it impossible to integrate $p_\theta(x)$ with an analytical solution. Thus, we cannot directly optimize this function as a result. We will discuss how to resolve this issue by approximating the unknown posterior distribution from only the observed data $x$ in the following section. 

### Autoencoders v.s. Variational Autoencoders

**Autoencoder** Before diving into Variational Autoencoders, let's take a look at **Autoencoder**, a model that encodes input by reconstructing the input itself. An Autoencoder contains an encoder and a decoder, with the goal of learning a low-dimensional feature representation from the input (unlabeled) training data. The encoder compresses the input data to low-dimensional feature vector z, while the decoder decomposes $z$ to the same shape as the input data.

The idea of Autoencoder is to *compress input images such that each vector in z contains meaningful factors of variation in data*. For example, if the inputs are different faces, the dimensions in z could be facial expressions, poses, different degrees of smile, etc.


However, we cannot generate new images from an autoencoder because we don't know the distributional space of z. VAE makes Autoencoders generative and allows us to sample from the model to generate data. VAE estimates the latent z representation so that we can generate more realistic images from the sampling. The intuition is z space shall reflect the factors of the variations. Assume that each image x is generated by sampling a new z with a slightly different factor of variations. Overall, z is used to conditionally generate the x.


### VAE Mathematical Explanation

We need two things to represent the model:
1. choose a proper $p(z)$:
Gaussian distribution is a reasonable choice for latent attributes. We can interpret every expression as a variation of the average neutral expression.
2. conditional distribution $p(x|z)$ is represented with the neural network:
We want to be able to generate a high-dimensional image from the simple low-dimension Gaussian distribution.

**Intractability** To train the model, we can learn model parameters to maximize likelihood of training data: $p_{\theta}(x) = \int p_{\theta}(z)p_{\theta}(x|z)dz$. However, this likelihood expression is intractable to evaluate or to optimize because we cannot compute $p(x|z)$ for every z in the intergral. We may also try to estimate posterior density $p_{\theta}(z|x) = p_{\theta}(x|z)p_{\theta}(z)/p_{\theta}(x)$ but it's also intractable due to the $p_{\theta}(x)$ term. Alternatively, in the paper, the authors proposed that we can approximate the true posterior $p_{\theta}(z|x)$ with $q_{\phi}(z|x)$, which is a lower bound on the data likelihood and can be optimized. 

The goal is to maximize the log-likelihood of $p_{\theta} (x^{(i)})$. Since $p_{\theta} (x^{(i)})$ does not depend on z, we can re-write as the expectation of z w.r.t $q_{\phi}(z|x^{(i)})$ and further derive (from lecture 12 slide):
$$
\begin{aligned}
\log p_{\theta} (x^{(i)}) &= \mathbb{E}_{z \sim q_{\phi}(z|x^{(i)})} [\log p_{\theta}(x^{(i)})] \\
&= \mathbb{E}_z [\log \frac{p_{\theta}(x^{(i)} | z)p_{\theta}(z)}{p_{\theta}(z | x^{(i)})}] \\
&= \mathbb{E}_{z} [\log \frac{p_{\theta}(x^{(i)} | z)p_{\theta}(z) q_{\phi}(z | x^{(i)}) }{p_{\theta}(z | x^{(i)}) q_{\phi}(z | x^{(i)}) }] \\
&= \mathbb{E}_z [\log p_{\theta} (x^{(i)} | z)] - \mathbb{E}_z [\log \frac{q_{\phi}(z | x^{(i)})}{p_{\theta}(z)}] + \mathbb{E}_z [\log \frac{q_{\phi}(z | x^{(i)})}{p_{\theta}(z | x_{(i)})}] \\
&=  \mathbb{E}_z [\log p_{\theta} (x^{(i)} | z)] - D_{KL}(q_{\phi}(z | x^{(i)}) || p_{\theta}(z)) + D_{KL}(q_{\phi}(z | x^{(i)})|| p_{\theta}(z | x^{(i)}))
\end{aligned}
$$

The estimate of $\mathbb{E}_z [\log p_{\theta} (x^{(i)} | z)]$ can be computed through sampling. $D_{KL}(q_{\phi}(z | x^{(i)}) || p_{\theta}(z))$ has closed-form solution. $D_{KL}(q_{\phi}(z | x^{(i)})|| p_{\theta}(z | x^{(i)}))$ would be always larger than or equal to zero. Thus, we have tractable lower bound. 

### VAE training process
![](https://i.imgur.com/IK8laIh.png)


$q_{\phi}(z | x^{(i)})$ is the encoder network in this process. The goal of $D_{KL}(q_{\phi}(z | x^{(i)}) || p_{\theta}(z))$ is to estimate a posterior distribution close to prior distribution. On the other hand, $p_{\theta} (x^{(i)} | z)$ is the decoder network and reconstruct the input data. We compute them in the forward pass for every minibatch of input data and then perform back-propagation. 


## Generative Adversarial Networks

### Overview of Generative Adversarial Networks
While implicit modeling is proven useful in generating data, it has the drawback of needing to estimate a probability distribution. What if we give up on explicitly modeling density, and just want the ability to sample? For **Generative Adversarial Networks**, we don't model the likelihood function $p(x)$ at all but only care about generating high-quality pictures.


From VAE, we learn that we can map a simple Gaussian distribution to a complex image distribution. We could leverage the same idea by mapping low dimensional noise to high dimensional image distribution. We can think of the decoder network as a generative network. The goal of Generative Adversarial Networks is to directly generate samples from a high-dimensional training distribution.


### Generative Adversarial Nets - 2014 original version
You may be curious: if we don't model z's distribution and don't know which sample z maps to which training image, how can we learn by reconstructing training images?

The general objective is to *generate images that should look "real"*. To achieve that, Generative Adversarial Net trains another network that learns to tell the difference between real and fake images and whether the generated image from a generator network looks like the one coming from the real distribution.


#### Discriminator network
The network that tells whether the image is real or fake is called the **Discriminator network**. We refer to images from the training distribution as real and the generated images from the generator network as fake. The discriminator network is essentially performing a supervised binary-class classification task. The discriminator uses the real/fake information to compute the gradient and backpropagate to the generation network, to make the generative examples more 'real'.

In the beginning, the discriminator can tell whether an image is a real input image or a generated image easily. Over time, as the generator network improves, images become more and more realistic. The discriminator has to change the decision boundary gradually to fit the new distribution better and better.

Python code example: 
```python
logits_real = D(real_data)

random_noise = sample_noise(batch_size, noise_size)
fake_images = G(random_noise)
logits_fake = D(fake_images.view(batch_size, 1, size, size))

d_total_error = discriminator_loss(logits_real, logits_fake)
```


#### Generator network
On the other side, the network that maps low dimensional noise to high dimensional image distribution is called the **Generator**. The goal of the generator is to fool the discriminator by generating real-looking images. In the beginning, the generator will generate random tensors that don't look like real images at all. However, the signal from the discriminator would inform the generator how it should improve the generated image to look more real. Over time, the generator would learn to generate more and more realistic samples.

python code example: 
```python
random_noise = sample_noise(batch_size, noise_size)
fake_images = G(random_noise)

gen_logits_fake = D(fake_images.view(batch_size, 1, size, size))
g_error = generator_loss(gen_logits_fake)
```

#### GAN Mathematical explanation 
Due to the coexistence of two networks, GAN is a two-player/min-max game that balances the optimization between Generator and Discriminator network. 

**Objective function**: 
$$
\begin{aligned}
\min_{\theta_g} \max_{\theta_d} [\mathbb{E}_{x \sim p_{data}} \log D_{\theta_d} (x) + \mathbb{E}_{z \sim p(z)} \log(1 - D_{\theta_d}(G_{\theta_g}(z)))]
\end{aligned}
$$

**Generator Objective**: $\min_{\theta_g}$ find weights that minimize this objective. $\mathbb{E}_{x \sim p_{data}} \log D_{\theta_d} (x)$ is the expectation of score predicted by discriminator, given on the training set. The generator would try to maximize this term because the generator tries to fool the discriminator by generating more realistic images.

**Discriminator Objective**: $\max_{\theta_d}$ find weights that maximize this objective.

During the training, generator transforms noise z to tensor and then the generated image is fed to the discriminator. Thus, $D_{\theta_d}(G_{\theta_g}(z))$ is the generated image score predicted by discriminator. Discriminator tries to tell the difference between real and fake images by moving this term to 0 and maximizing $1 - D_{\theta_d}(G_{\theta_g}(z))$. 


The training process is to alternate between 
1. Gradient ascent on discriminator 
2. Gradient descent on generator

Problem of 2 is the gradient dominated by the region sample is already good. Training is very slow and unstable at the beginning. One solution is to change to use gradient ascent on generator and modify the different objective.

![](https://i.imgur.com/zvAex6j.png) 
<figcaption> Generative Adversaial Nets training flow </figcaption>


### Evaluation

Recall that there are two objectives in Generative Modeling: 
1. Learn $p_{model} (x)$ that approximates $p_{data}(x)$
2. Sampling new data $x$ from $p_{model}(x)$ 

When evaluating the GAN output, we want to make sure the two objectives are taken care of.


#### Inception Scores
Inception score was a popular evaluation metric, which evaluates the quality of generated images. Inception Score uses the Inception V3 pre-trained model on ImageNet to observe the distribution of generated images. If the generated image is easily recognized by the discriminator, the classification score (i.e. $p(y|x)$) would be large, which leads to $p(y|x)$ having low entropy. Meanwhile, we also want the marginal distribution $p(y)$ to have high entropy. The inception score can be calculated as follows:
$$
\begin{aligned}
IS(x) &= \exp(\mathbb{E}_{x \sim p_g}[D_{KL}[p(y|x) || p(y)]]) \\
&= \exp(\mathbb{E}_{x \sim p_g, y \sim p(y | x)} [\log p(y | x) - \log p(y)]) \\
&= \exp(H(y) - H(y | x))
\end{aligned}
$$
where a high inception score indicates better-quality generated images. However, the inception score has some drawbacks and is fooled over the years. Thus, people started using measurements such as FID in recent years. 

#### Nearest Neighbours
A simpler evaluation method is to visualize a sample of generated images to tell how realistic the generated images are. We can also leverage Nearest Neighbours to compare real images and generated images. The idea is to sample some real images from the training set and calculate the distance between the sampled generated images. If the generated images are real-looking, the distances should be small.

#### HYPE - Human eye perceptual Evaluation 
HYPE is a new evaluation method introduced in 2019. It evaluates GAN by a social computing method: the website invites users to evaluate GAN and try to build metrics on top of it. The goal is to ensure the evaluation is consistent while evaluating different types of GANs.


### Challenges

#### Optimization 
It's not easy to train GAN because the process has many challenges. Often times, the generator and discriminator loss keeps oscillating during GAN training. There is also no stopping criterion in practice. Also, when the discriminator is very confidently classifying fake samples, the generator training may fail due to vanishing gradients.  



#### Mode collapse

Mode collapse happens when the generator learns to fool the discriminator by producing a single class from the whole training dataset. Often time the training dataset is multi-modal, which means the probability density distribution over features has multiple peaks. If data is imbalanced or some other problems happen during the training process, the generating image may collapse into one mode or few modes while other modes are disappearing. For example, the discriminator classifies a lot of generated images incorrectly. The generator takes the feedback and only generates images that are the same or similar to the ones that fool the discriminator. Eventually, the generated images collapse into single-mode or fewer modes.

### Case Studies

#### DCGAN
The idea of DCGAN is to use a convolutional neural network in GAN. Here are some architecture guidelines DCGAN gave in their paper:
> 1. Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
> 2. Use batch norm in both the generator and the discriminator.
> 3. Remove fully connected hidden layers for deeper architectures
> 4. Use ReLU activation in generator for all layers except for the output, which uses Tanh.
> 5. Use LeakyReLU activation in the discriminator for all layers.


#### CycleGAN
Image-to-image translation is a class of problems where the goal is to map an image to another image within the same pair. For example, one may wish to map an image of a location during the Spring season to an image of the same location but during the Fall season. However, paired images are not always availabe. The goal of CycleGAN is then to learn a mapping $G$ such that the distribution $G(X)$ is as similar to the distribution of $Y$ as possible, where $X$ and $Y$ are the input and output images respectively. 

#### StyleGAN
StyleGAN is an extension of GAN that aims to improve the generator's ability to generate a wider variety of images. The main modifications to the architecture of GAN's generator is by having two sources of randomness (instead of one): a mapping network that controls the style of the output image, and an additional noise that adds variability to the image. Applications of StyleGAN include human-face generation, anime character generations, new fonts, etc. 


### Summary

Comparision between the methods


|  |  Pixel RNN/CNN   | Variational AutoEncoders | Generative Adversial Modeling |
| -------- | --- | -------- | -------- |
|    Pros      | <ul><li>Can explicitly compute likelihood p(x) </li> <li> Easy to optimize </li> <li>Good samples </li></ul>| <ul><li>principled approach to generative models </li> <li>interpretable latent space </li> <li> allows inference of q(z\|x) </li> <li>can be useful feature representation for other tasks  </li></ul>    | Beautiful, state-of-the-art samples!         |
|    Cons     | slow sequential generation    | <ul><li>Maximizes lower bound of likelihood which is not as good evaluation as PixelRNN/PixelCNN </li> <li> Samples blurrier and lower quality compared to state-of-the-art (GANs) </li></ul>    | <ul><li>Trickier/more unstable to train </li> <li> cannot solve inference queries such as p(x) p(z\|x)  </li></ul>   |


### Further Reading
These readings are optional and contain pointers of interest. 
> PixelRNN/CNN: https://arxiv.org/pdf/1601.06759.pdf
> Variational Auto-Encoders: https://arxiv.org/pdf/1312.6114.pdf
> Generative Adversial Net: https://arxiv.org/pdf/1406.2661.pdf
> DCGAN: https://arxiv.org/pdf/1511.06434.pdf
> CycleGAN: https://arxiv.org/pdf/1703.10593.pdf
> StyleGAN: https://arxiv.org/pdf/1812.04948.pdf
> Mode collapse: https://www.coursera.org/lecture/build-basic-generative-adversarial-networks-gans/mode-collapse-Terkm
> HYPE: https://arxiv.org/pdf/1904.01121.pdf

