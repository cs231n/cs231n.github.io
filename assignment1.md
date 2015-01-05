---
layout: page
mathjax: true
permalink: /assignment1/
---

In this assignment you will practice putting together a simple image classification pipeline, based on the k-Nearest Neighbor or the SVM/Softmax classifier. The goals of this assignment are as follows:

- understand the basic **Image Classification pipeline** and the data-driven approach (train/predict stages)
- understand the train/val/test **splits** and the use of validation data for **hyperparameter tuning**.
- develop proficiency in writing efficient **vectorized** code with numpy
- implement and apply a k-Nearest Neighbor (**kNN**) classifier
- implement and apply a Multiclass Support Vector Machine (**SVM**) classifier
- implement and apply a **Softmax** classifier
- understand the differences and tradeoffs between these classifiers
- get a basic understanding of performance improvements from using **higher-level representations** than raw pixels (e.g. color histograms, Histogram of Gradient (HOG) features)

Stanford students taking the class: See the [assignment page](http://vision.stanford.edu/teaching/cs231n/assignments.html) for further details.

### Q1: k-Nearest Neighbor classifier (30 points)

The IPython Notebook **knn.ipynb** will walk you through implementing the kNN classifier.

### Q2: Training a Support Vector Machine (30 points)

The IPython Notebook **svm.ipynb** will walk you through implementing the SVM classifier.

### Q3: Implement a Softmax classifier (30 points)

The IPython Notebook **softmax.ipynb** will walk you through implementing the Softmax classifier.

### Q4: Higher Level Representations: Image Features (10 points)

The IPython Notebook **features.ipynb** will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.

### Q5: Bonus: Design your own features! (+10 points)

In question 4 we provided you with some features. For bonus points, implement your own additional features from scratch using only numpy or scipy (no external dependencies). You may have to research different feature types to get ideas for what you might want to implement. To get the bonus points, concatenating your new feature to the old features should improve your performance beyond what you got in Q4.

### Q6: Cool Bonus: Do something extra! (+10 points)

When completing assignments, your task as a student is usually to answer a set of questions we wrote. In this Cool Bonus meta question, we'd like to flip this and encourage you to make up and answer your own questions regarding the material. 

To claim these points implement, investigate and analyze something related to the topics in this assignment, using the code base and data in this assignment. For example, is there some other interesting question we could have asked? Pose it, explore it and answer it. Or is there any insightful visualization you can make? Plot it and interpret it. Or maybe you can devise a new experiment, such as a different spin on the loss function? If you try out something fun and get some cool results you'll get the points, and **we will feature the coolest results in the lecture**.
