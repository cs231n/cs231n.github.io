Table of contents:

##### Adversarial Attacks
- Poisoning Attack vs. Evasion Attack
- White-box Attack vs. Black-box Attack
- Adversarial Goals

##### Adversarial Examples
- Fast Gradient Sign Method (FGSM)
- One-Pixel Attack

##### Defense Strategies against Adversarial Examples
- Adversarial Training
- Defensive Distillation
- Denoising

## Adversarial Attacks
In class, we have seen examples where image classification models can be fooled with adversarial examples. By adding small and often imperceptible perturbations to images, these adversarial attacks can deceive a deep learning model to label the modified image as a completely different class. In this note, we will give an overview of the various types of adversarial attacks on machine learning models, discuss several representative methods for generating adversarial examples, as well as some possible defense methods and mitigation strategies against such adversarial attacks.

**Adversary**. In computer security, the term “adversary” refers to people or machines that attempt to penetrate or corrupt a computer network or system. In the context of machine learning and deep learning, adversaries can use a variety of attack methods to disrupt a machine learning model, and cause it to behave erratically (e.g. to misclassify a dog image as a cat image). In general, attacks can happen either during model training (known as a “poisoning” attack) or after the model has finished training (an “evasion” attack).

### Poisoning Attack vs. Evasion Attack
**Poisoning attack**. A poisoning attack involves polluting a machine learning model's training data. Such attacks take place during the training time of the machine learning model, when an adversary presents data that is intentionally mislabeled to the model, therefore instilling misleading knowledge, and will eventually cause the model to make inaccurate predictions at test time. Poisoning attacks require that an adversary has access to the model’s training data, and is able to inject misleading data into the training set. To make the attacks less noticeable, the adversary may decide to slowly introduce ill-intentioned samples over an extended period of time.

An example of a poisoning attack took place in 2016, when Microsoft launched a Twitter chatbot, Tay. Tay was designed to mimic the language patterns of a 18- to 24- year-old in the U.S. for entertainment purposes, to engage people through “casual and playful conversation”, and to learn from its conversations with human users on Twitter. However, soon after its launch, a vulnerability in Tay was exploited by some adversaries, who interacted with Tay using profane and offensive language. The attack caused the chatbot to learn and internalize inappropriate language. The more Tay engaged with adversarial users, the more offensive Tay’s tweets became. As a result, Tay was quickly shut down by Microsoft, only 16 hours after its launch.

**Evasion attack**. In evasion attacks, the adversary tries to evade or fool the system by adjusting malicious samples during the testing phase. Compared with poisoning attacks, evasion attacks are more common, and easier to conduct. One main reason is that with evasion attacks, adversaries don’t necessarily need to access the training data, nor inject bad data into the training process of the model.

### White-box vs. Black-box Attacks
The evasion attacks discussed above occur during the testing phase of the model. The effectiveness of such attacks depends on the amount of information available to the adversary about the model. Before we dive into the various methods for generating adversarial examples, let’s first briefly discuss the differences between white-box attacks and black-box attacks, and understand the various adversarial goals.

**White-box attack**. In a white-box attack, the adversary is assumed to have total knowledge about the model, such as the model architecture, number of layers, the weights of the final trained model, etc. The adversary also has knowledge on the model’s training process, such as the optimization algorithm (e.g. Adam, RMSProp etc.) that is used, the data that the model is trained on, the distribution of the training data, and the model’s performance on the training data. It can be very dangerous if the adversary is able to identify the feature space where the model has a high error rate, and use that information to construct adversarial examples and exploit the model. The more the adversary knows, the more severe the attacks can be, and so are the consequences.

**Black-box attack**. On the contrary, in black-box attacks, the adversary assumes no knowledge about the model. Instead of constructing adversarial examples based on prior knowledge, the adversary exploits a model by providing a series of carefully crafted inputs and observing outputs. Through trial and error, the attacks may eventually be successful in misleading the model to make the wrong predictions.

### Adversarial Goals
The goals of the adversarial attacks can be broadly categorized as follows:
- **Confidence Reduction**. The adversary aims to reduce the model’s confidence in its predictions, which does not necessarily lead to the wrong class output. For example, due to the adversarial attack, a model which originally classifies an image of a cat with high probability ends up outputting a lower probability for the same image and class pair.
- **Untargeted Misclassification**. The adversary tries to misguide the model to predict any of the incorrect classes. For example, when presented with an image of a cat, the model outputs any class that is non-cat (e.g. dog, airplane, computer, etc.).
- **Targeted Misclassification**. The adversary tries to misguide the model to output a particular class other than the true class. For example, when presented with an image of a cat, the model is forced to classify it as a dog image, where the output class of dog is specified by the adversary.

Generally speaking, targeted attacks are more sophisticated than untargeted attacks, which are in turn more difficult than confidence reduction.

## Adversarial Examples
In CS231n, we mainly focus on examples of adversarial images in the context of image classification. In an adversarial attack, the adversary attempts to modify the original input image by adding some carefully crafted perturbations, which can cause the image classification model to yield mispredictions. Oftentimes, the generated perturbations are either too small to be visually identified by human eyes, or small enough that humans consider them to be harmless, random noise. And yet, these perturbations can be “meaningful” and misleading to the image classification model. Below, we discuss two methods to generate adversarial examples.

<p align="center">
  <img src="/assets/adversary-1.png" width="450">
  <div class="figcaption">An example of adversarial attack, in which a tiny amount of carefully crafted perturbations leads to misclassification. Here the perturbations are so small that they only become visible to humans after being magnified for about 30 times.</div>
</p>


### Fast Gradient Sign Method (FGSM)
The simplest yet highly efficient algorithm for generating adversarial examples is known as the Fast Gradient Sign Method (FGSM), which is a single step attack on images. Proposed by Goodfellow et al. in 2014, FGSM combines a white box approach with a misclassification goal. Using FGSM, a small perturbation is first generated in the direction of the sign of the gradients with respect to the input image. Next, the generated perturbations are added to the original image, resulting in an adversarial image. The equation for untargeted attack using FGSM is given by:

$$ adv\_x = x + \epsilon*\text{sign}(\nabla_xJ(\theta, x, y)) $$

Here, $$ J $$ is the cost function (e.g. cross-entropy cost) of the trained model, $$ \nabla_x $$ denotes the gradient of the model’s loss function with respect to the original image $$ x $$. The thing to note here is we are calculating the gradient with respect to the pixels of the image. From the gradient of the model’s loss function, we take the sign of each term in the gradient, reducing it to a matrix of 1s, 0s and -1s. The intuition here is that we nudge the pixels of the image in the direction that maximizes the loss. In other words, we perform **gradient ascent** instead of gradient descent, since the goal is to increase the error and let the model output the incorrect results.

Having obtained the sign of the gradient, we then multiply the result with a tiny value, ϵ, which controls the amount of perturbations (i.e. the perturbation’s amplitude) to be added. The larger the value of epsilon, the more noticeable the perturbations are to humans. Recall that from the adversary’s perspective, the goal is to ensure the corruption to the original image is imperceptible, while being able to fool the classification model. ϵ is a hyper-parameter to be chosen.

FGSM can also be used for targeted misclassification attacks. In this case, the adversary aims to maximize the probability of some specific target class, which is unlikely to be the true class of the original image $$ x $$:

$$ adv\_x = x - \epsilon*\text{sign}(\nabla_xJ(\theta, x, y_{target})) $$

The difference is in case of targeted attacks, we minimize the loss between the model’s predicted class and the target class, whereas in case of untargeted attack we maximize the loss instead of minimize it.

In addition to only applying the FGSM equation once, a straightforward extension is to develop an iterative procedure, and run FGSM multiple times. Here is what the iterative procedure might look like for untargeted attacks using FGSM, when implemented with TensorFlow:

```
import tensorflow as tf
import keras.backend as K

# Get the true label of the iamge
correct_label = get_correct_label()
total_class_count = N

# Initialize adversarial example with original input image
x_adv = original_img
x_adv = tf.convert_to_tensor(x_adv, dtype=tf.float32)

# Initialize the perturbations
noise = np.zeros_like(original_img)

# Epsilon is a hyper-parameter
epsilon = 0.01
epochs = 100

for i in range(epochs):
    target = K.one_hot(correct_label, total_class_count)

    with tf.GradientTape() as tape:
        tape.watch(x_adv)
        prediction = model(x_adv)
        loss = K.categorical_crossentropy(target, prediction[0])

    # Calculate the gradient
    grads = tape.gradient(loss, x_adv)

    # Get the sign of the gradient
    delta = K.sign(grads[0])
    noise = noise + delta

    # Generate an adversarial example with FGSM
    x_adv = x_adv + epsilon*delta

    # Get the latest model output
    preds = model.predict(x_adv, steps=1).squeeze()
    pred = np.argmax(preds, axis=-1)

    # Exit the procedure if model is fooled
    if pred != correct_label:
      break
```

In the example implementation above, we also employ early-stopping and exit the iterative procedure once the model is fooled. This helps minimize the amount of perturbations added, and may also improve the efficiency by reducing the time needed to fool the model. With the iterative approach, we can also obtain additional adversarial examples when the procedure is run for more iterations.

<p align="center">
  <img src="/assets/adversary-2.png" width="550">
  <div class="figcaption">An example of a “successful” adversarial attack in which the image classifier recognized a watermelon as a tomato. In this case, although the goal of misclassification is achieved, the unmagnified perturbations are large enough to be perceived by human eyes.</div>
</p>

In practice, FGSM attacks work particularly well for network architectures that favor linearity, such as logistic regression, maxout networks, LSTMs, networks that use the ReLU activation function, etc. While ReLU is non-linear, when ϵ is sufficiently small, the ReLU activation does not change the sign of the gradient with respect to the original image, and thus will not prevent the pixels of the image to be nudged in the direction that maximizes the loss. The authors of FGSM stated that changing to nonlinear model families such as RBF networks confer a significant reduction in a model’s vulnerability to adversarial examples.

### One-Pixel Attack
In order to fool a machine learning model, the Fast Gradient Sign Method discussed above requires many pixels of the original image to be changed, if only by a little. As shown in the example image above, sometimes the modifications might be excessive (i.e. the amount of modified pixels are fairly large) such that they become visually identifiable to human eyes. One may then wonder if it’s possible to modify fewer pixels, while still keeping the model fooled? The answer is yes. In 2019, a method for generating one-pixel adversarial perturbations was proposed, in which an adversarial example can be generated by modifying just one pixel.

The One-pixel attack uses differential evolution to find out which pixel is to be changed, and how. Differential evolution (DE) is a type of evolutionary algorithm (EA). It is a population based optimization algorithm for solving complex optimization problems. In specific, during each iteration, a set of candidate solutions (children) is generated according to the current population (parents). The candidate solutions are then compared with their corresponding parents, surviving if they are better candidate solutions according to some criterion. The process repeats until some stopping criterion are met.

In one-pixel attack, each candidate solution encodes a pixel modification and is represented by a vector of five elements: the x and y coordinates, and the red, green and blue (RGB) values of the pixel. The search starts with 400 initial candidate solutions. In each iteration, another 400 candidate solutions (children) are generated using the following formula:

$$ x_{i}(g+1) = x_{r1}(g) + F(x_{r2}(g) - x_{r3}(g)), $$

$$ r1 \neq r2 \neq r3 $$

where $$ x_{i} $$ is an element of the candidate solution, $$ g $$ is the current generation, $$ F $$ is the scale parameter set to be 0.5, and $$ r1 $$, $$ r2 $$, $$ r3 $$ are different random numbers. The search stops when one of the candidate solutions is an adversarial example that fools the model successfully, or if the maximum number of iterations specified has been reached.

By using differential evolution, one-pixel attack has several advantages. Since DE doesn’t use gradient information for optimization, it’s not required for the objective function to be differentiable, as is the case with classical optimization methods such as gradient descent. Calculating the gradient requires much more information about the model to be exploited, therefore not needing gradient information makes conducting the attack more feasible. Finally, it’s worth noting that one-pixel attack is a type of black-box attack, which assumes no information about the classification model; it is sufficient to just observe the model’s output probabilities.

## Defense Strategies against Adversarial Examples
Having discussed some techniques for generating adversarial examples, we now turn our attention to possible defense strategies against such adversarial attacks. While we go through each of the countermeasures, it’s worth keeping in mind that none of them can act as a panacea for all challenges. Moreover, implementing such defense strategies may incur extra performance costs.

### Adversarial Training
One of the most intuitive and effective defenses against adversarial attacks is adversarial training. The idea of adversarial training is to incorporate adversarial samples into the model training stage, and thus increase model robustness. In other words, since we know that the original training process leads to models that are vulnerable to adversarial examples, we just also train on adversarial examples so that the models have some “immunity” over the adversarial examples.

To perform adversarial training, the defender simply generates a lot of adversarial examples and include them in the training data. At training time, the model is trained to assign the same label to the adversarial example as to the original example. For example, upon seeing an adversarially perturbed training image, whose original label is cat, the model should learn the correct label for the perturbed image is still cat.

The problem with adversarial training is that it is only effective in defending the models against the same attacks used to craft the examples originally included in the training pool. In black-box attacks, adversaries only need to find one crack in a system’s defenses for an attack to go through. It can be likely that the attack method employed by the adversary is not anticipated by the defender at the time of model training, therefore leaving the adversarially trained model vulnerable to the unseen attacks.

### Defensive Distillation
Introduced in 2015 by Papernot et al., defensive distillation uses the idea of distillation and knowledge transfer to reduce the effectiveness of adversarial samples on deep neural networks. The term distillation was originally proposed as a way to transfer knowledge from a large neural network to a smaller one. Doing so can help reduce the computational complexity of deep neural networks, and facilitate the deployment of deep learning models in resource constrained devices. In defensive distillation, instead of transferring knowledge between models of different architectures, the knowledge is extracted from a model to then improve its own resilience to adversarial examples.

Let's assume we are training a neural network for image classification tasks, and the network is designed with a softmax layer as the output layer. The key point in distillation is the addition of a **temperature parameter T** to the softmax operation:

$$ F(X) = \left[ \frac{e^{z_i(x)/T}}{\sum_{l=0}^{N-1} e^{z_l(x)/T}} \right]_{i \in 0 ... N-1} $$

The authors showed that experimentally, a high empirical value of T gives a better distillation performance. During test time, T is set to 1, making the above equation equivalent to standard softmax operation.

Defensive distillation is a two-step process. First, we train an initial network $$F$$ on data $$X$$. In this step, instead of letting the network output hard class labels, we take the probability vectors produced by the softmax layer. The benefit of using class probabilities (i.e. soft labels) instead of hard labels is that in addition to merely providing a sample’s correct class, probabilities also encode the relative differences between classes. Next, we then use the probability vectors from the initial network as the labels to train another distilled network $$F’$$ with the same architecture on the same training data $$X$$. During training, it is important to set the temperature parameter $$T$$ for both networks to a value larger than 1. After training is completed, we will use the distilled network $$F’$$ with $$T$$ set to 1 to make predictions at test time.

<p align="center">
  <img src="/assets/defensive-distillation.png" width="600">
  <div class="figcaption">Defense mechanism based on a transfer of knowledge contained in probability vectors through distillation.</div>
</p>

Why is defensive distillation a good idea? First, a large value of $$T$$ has the effect of pushing the resulting probability distribution closer to uniform. This helps improve the model’s ability to generalize outside of its training dataset, by avoiding situations where the model is forced to make an overly confident prediction in one class when a sample includes characteristics of two or more classes. The authors also argue that distillation at high temperatures reduces a model’s sensitivity to small input variations, which are often found in adversarial examples. The model’s sensitivity to input variation is quantified by its Jacobian:

$$ \frac{\partial F_i(X)}{\partial X_j} = \frac{\partial}{\partial X_j}  \left( \frac{e^{z_i/T}}{\sum_{l=0}^{N-1} e^{z_l/T}} \right) \\
= \frac{1}{T} \frac{e^{z_i/T}}{g^2(X)} \left( \sum_{l=0}^{N-1} \left(\frac{\partial z_i}{\partial X_j} - \frac{\partial z_l}{\partial X_j} \right) e^{z_l/T} \right) $$

where

$$ g(X) = \sum_{l=0}^{N-1} e^{z_l(X)/T} $$

From the above expression, it can be observed that the amplitude of the Jacobian is inversely proportional to the temperature value. During test time, although $$T$$ is set to a relatively small value of 1, the model’s sensitivity to small variations and perturbations will not be affected, since the weights learned at training time remain unchanged, and decreasing temperature only makes the class probability vector more discrete without changing the relative ordering of the classes.

### Denoising
Since adversarial examples are images with added perturbations (i.e. noise), one straightforward defense strategy is to have some mechanisms to denoise the adversarial samples. There can be two approaches to denoising: input denoising and feature denoising. Input denoising attempts to partially or fully remove the adversarial perturbations from the input images, whereas feature denoising aims to alleviate the effects of adversarial perturbations on high-level features.

In their study, Chow et al. proposed a method for input denoising with ensembles of denoisers. The intuition of using ensembles for denoising is that there are various ways for adversaries to generate and add perturbations to images, and no single denoiser is guaranteed to be effective across all data corruption methods - a denoiser that excels at removing some types of noise may perform poorly on others. Therefore, it is often helpful to employ an ensemble of diverse denoisers, instead of relying only on a single denoiser.

Autoencoders are used for training the denoisers. First, the denoising autoencoder takes a clean input image and transforms it into an adversarial example by adding some perturbations. Next, the noisy image is fed into the auto encoder, with a goal of reconstructing the original clean, uncorrupted image. Given N training examples, the denoising autoencoder is trained by backpropagation to minimize the reconstruction loss:

$$ Loss = \frac{1}{N}\sum_{i=1}^{N}d(x_i, g_{\theta'}(f_{\theta}(x'_i))) + \frac{\lambda}{2}(\|\theta \|_\text{F}^2 + \|\theta' \|_\text{F}^2) $$

where d is a distance function and λ is a regularization hyperparameter penalizing the Frobenius norm of θ and θ’. $$ g^i $$ is the operation at the i-th decoding layer with weights $$ \theta_i^' $$.

For feature denoising as a defense strategy, one study was conducted by Xie et al, in which the authors incorporated denoising blocks at intermediate layers of a convolutional neural network. The authors argued that the adversarial perturbation of the features gradually increases as an image is propagated through the network, causing the model to eventually make the wrong predictions. Therefore, it can be helpful to add denoising blocks at intermediate layers of the network to combat feature noise.

<p align="center">
  <img src="/assets/denoising-block.png" width="300">
  <div class="figcaption">Defense mechanism based on a transfer of knowledge contained in probability vectors through distillation.</div>
</p>

The input to a denoising block can be any feature layer in the convolutional neural network. In the study, each denoising blocks performs one type of the following denoising operations: nonlocal means, bilateral filter, mean filter, and median filter. These are the techniques commonly used in computer vision tasks such as image processing and denoising. The denoising blocks are trained jointly with all layers of the network in an end-to-end manner using adversarial training. In their experiments, denoising blocks were added to the variants of ResNet models. The results showed that the proposed denoising method achieved 55.7 percent accuracy under white-box attacks on ImageNet, whereas previous state of the art was only 27.9 percent accuracy.

## References
[Explaining and harnessing adversarial examples](https://arxiv.org/abs/1412.6572)  
[One pixel attack for fooling deep neural networks](https://arxiv.org/abs/1710.08864)  
[Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks](https://arxiv.org/abs/1511.04508)  
[Denoising and Verification Cross-Layer Ensemble Against Black-box Adversarial Attacks](https://arxiv.org/abs/1908.07667)  
[Feature Denoising for Improving Adversarial Robustness](https://arxiv.org/abs/1812.03411)  
[Adversarial Attacks and Defences: A Survey](https://arxiv.org/abs/1810.00069)

## Additional Resources
[Adversarial Robustness - Theory and Practice (NeurIPS 2018 tutorial)](https://adversarial-ml-tutorial.org/)  
[CleverHans - An Python Library on Adversarial Example](https://github.com/cleverhans-lab/cleverhans)
