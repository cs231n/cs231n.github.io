---
layout: page
title: Assignment 2
mathjax: true
permalink: /assignments2021/assignment2/
---

<span style="color:red">This assignment is due on **Friday, April 30 2021** at 11:59pm PST.</span>

Starter code containing Colab notebooks can be [downloaded here]({{site.hw_2_colab}}).

- [Setup](#setup)
- [Goals](#goals)
- [Q1: Multi-Layer Fully Connected Neural Networks (16%)](#q1-multi-layer-fully-connected-neural-networks-16)
- [Q2: Batch Normalization (34%)](#q2-batch-normalization-34)
- [Q3: Dropout (10%)](#q3-dropout-10)
- [Q4: Convolutional Neural Networks (30%)](#q4-convolutional-neural-networks-30)
- [Q5: PyTorch/TensorFlow on CIFAR-10 (10%)](#q5-pytorchtensorflow-on-cifar-10-10)
- [Submitting your work](#submitting-your-work)

### Setup

Please familiarize yourself with the [recommended workflow]({{site.baseurl}}/setup-instructions/#working-remotely-on-google-colaboratory) before starting the assignment. You should also watch the Colab walkthrough tutorial below.

<iframe style="display: block; margin: auto;" width="560" height="315" src="https://www.youtube.com/embed/IZUz4pRYlus" frameborder="0" allowfullscreen></iframe>

**Note**. Ensure you are periodically saving your notebook (`File -> Save`) so that you don't lose your progress if you step away from the assignment and the Colab VM disconnects.

While we don't officially support local development, we've added a <b>requirements.txt</b> file that you can use to setup a virtual env.

Once you have completed all Colab notebooks **except `collect_submission.ipynb`**, proceed to the [submission instructions](#submitting-your-work).

### Goals

In this assignment you will practice writing backpropagation code, and training Neural Networks and Convolutional Neural Networks. The goals of this assignment are as follows:

- Understand **Neural Networks** and how they are arranged in layered architectures.
- Understand and be able to implement (vectorized) **backpropagation**.
- Implement various **update rules** used to optimize Neural Networks.
- Implement **Batch Normalization** and **Layer Normalization** for training deep networks.
- Implement **Dropout** to regularize networks.
- Understand the architecture of **Convolutional Neural Networks** and get practice with training them.
- Gain experience with a major deep learning framework, such as **TensorFlow** or **PyTorch**.

### Q1: Multi-Layer Fully Connected Neural Networks (16%)

The notebook `FullyConnectedNets.ipynb` will have you implement fully connected
networks of arbitrary depth. To optimize these models you will implement several
popular update rules.

### Q2: Batch Normalization (34%)

In notebook `BatchNormalization.ipynb` you will implement batch normalization, and use it to train deep fully connected networks.

### Q3: Dropout (10%)

The notebook `Dropout.ipynb` will help you implement dropout and explore its effects on model generalization.

### Q4: Convolutional Neural Networks (30%)

In the notebook `ConvolutionalNetworks.ipynb` you will implement several new layers that are commonly used in convolutional networks.

### Q5: PyTorch/TensorFlow on CIFAR-10 (10%)

For this last part, you will be working in either TensorFlow or PyTorch, two popular and powerful deep learning frameworks. **You only need to complete ONE of these two notebooks.** While you are welcome to explore both for your own learning, there will be no extra credit.

Open up either `PyTorch.ipynb` or `TensorFlow.ipynb`. There, you will learn how the framework works, culminating in training a convolutional network of your own design on CIFAR-10 to get the best performance you can.

### Submitting your work

**Important**. Please make sure that the submitted notebooks have been run and the cell outputs are visible.

Once you have completed all notebooks and filled out the necessary code, you need to follow the below instructions to submit your work:

**1.** Open `collect_submission.ipynb` in Colab and execute the notebook cells.

This notebook/script will:

* Generate a zip file of your code (`.py` and `.ipynb`) called `a2.zip`.
* Convert all notebooks into a single PDF file.

If your submission for this step was successful, you should see the following display message:

`### Done! Please submit a2.zip and the pdfs to Gradescope. ###`

**2.** Submit the PDF and the zip file to [Gradescope](https://www.gradescope.com/courses/257661).

Remember to download `a2.zip` and `assignment.pdf` locally before submitting to Gradescope.
