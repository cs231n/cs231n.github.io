---
layout: page
title: Assignment 3
mathjax: true
permalink: /assignments2025/assignment3/
---

<span style="color:red">This assignment is due on **Friday, May 30 2025** at 11:59pm PST.</span>

Starter code containing Colab notebooks can
be [downloaded here](https://drive.google.com/file/d/14J1IBXY50431YBbOWmPpYngudbnIqUFP/view?usp=drive_link).

- [Setup](#setup)
- [Goals](#goals)
- [Q1: Image Captioning with Transformers](#q1-image-captioning-with-transformers)
- [Q2: Self-Supervised Learning for Image Classification](#q2-self-supervised-learning-for-image-classification)
- [Q3: Denoising Diffusion Probabilistic Models](#q3-denoising-diffusion-probabilistic-models)
- [Q4: CLIP and Dino](#q4-clip-and-dino)
- [Submitting your work](#submitting-your-work)

### Setup

Please familiarize yourself with
the [recommended workflow]({{site.baseurl}}/setup-instructions/#working-remotely-on-google-colaboratory) before starting
the assignment. You should also watch the Colab walkthrough tutorial below.

<iframe style="display: block; margin: auto;" width="560" height="315" src="https://www.youtube.com/embed/DsGd2e9JNH4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

**Note**. Ensure you are periodically saving your notebook (`File -> Save`) so that you don't lose your progress if you
step away from the assignment and the Colab VM disconnects.

While we don't officially support local development, we've added a <b>requirements.txt</b> file that you can use to
setup a virtual env.

Once you have completed all Colab notebooks **except `collect_submission.ipynb`**, proceed to
the [submission instructions](#submitting-your-work).

### Goals

In this assignment, you will implement language networks and apply them to image captioning on the COCO dataset. Then
you will be introduced to self-supervised learning to automatically learn the visual representations of an unlabeled
dataset. Next, you will implement diffusion models (DDPMs) and apply them to image generation. Finally, you will explore
CLIP and DINO, two self-supervised learning methods that leverage large amounts of unlabeled data to learn visual
representations.

The goals of this assignment are as follows:

- Understand and implement Transformer networks. Combine them with CNN networks for image captioning.
- Understand how to leverage self-supervised learning techniques to help with image classification tasks.
- Implement and understand diffusion models (DDPMs) and apply them to image generation.
- Implement and understand CLIP and DINO, two self-supervised learning methods that leverage large amounts of unlabeled
  data to learn visual representations.

**You will use PyTorch for the majority of this homework.**

### Q1: Image Captioning with Transformers

The notebook `Transformer_Captioning.ipynb` will walk you through the implementation of a Transformer model and apply it
to image captioning on COCO.

### Q2: Self-Supervised Learning for Image Classification

In the notebook `Self_Supervised_Learning.ipynb`, you will learn how to leverage self-supervised pretraining to obtain
better performance on image classification tasks. **When first opening the notebook, go
to `Runtime > Change runtime type` and set `Hardware accelerator` to `GPU`.**

### Q3: Denoising Diffusion Probabilistic Models

In the notebook `DDPM.ipynb`, you will implement a Denoising Diffusion Probabilistic Model
(DDPM) and apply it to image generation.

### Q4: CLIP and Dino

In the notebook `CLIP_DINO.ipynb`, you will implement CLIP and DINO, two self-supervised learning methods that leverage
large amounts of unlabeled data to learn visual representations.

### Submitting your work

**Important**. Please make sure that the submitted notebooks have been run and the cell outputs are visible.

Once you have completed all notebooks and filled out the necessary code, you need to follow the below instructions to
submit your work:

**1.** Open `collect_submission.ipynb` in Colab and execute the notebook cells.

This notebook/script will:

* Generate a zip file of your code (`.py` and `.ipynb`) called `a3_code_submission.zip`.
* Convert all notebooks into a single PDF file called `a3_inline_submission.pdf`.

If your submission for this step was successful, you should see the following display message:

`### Done! Please submit a3_code_submission.zip and a3_inline_submission.pdf to Gradescope. ###`

**2.** Submit the PDF and the zip file to Gradescope.

Remember to download `a3_code_submission.zip` and `a3_inline_submission.pdf` locally before submitting to Gradescope.
