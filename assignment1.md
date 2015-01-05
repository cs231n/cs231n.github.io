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

**Deliverables:** The assignment is spread across four IPython Notebooks. Complete, print and hand in your IPython notebooks. The IPython Notebooks will also ask you to edit external files in the *cs231n* folder: print and hand in all of these files as well.

### Q1: k-Nearest Neighbor classifier (30 points)

The IPython Notebook **knn.ipynb** will walk you through implementing the kNN classifier.

### Q2: Training a Support Vector Machine (30 points)

The IPython Notebook **svm.ipynb** will walk you through implementing the SVM classifier.

### Q3: Implement a Softmax classifier (30 points)

The IPython Notebook **softmax.ipynb** will walk you through implementing the Softmax classifier.

### Q4: Higher Level Representations: Image Features (10 points)

The IPython Notebook **features.ipynb** will walk you through this exercise, in which you will examine the improvements gained by using higher-level representations as opposed to using raw pixel values.

### Q5: Bonus: Design your own features! (+10 points)

In this assignment we provide you with Color Histograms and HOG features. To claim these bonus points, implement your own additional features from scratch, and using only numpy or scipy (no external dependencies). You will have to research different feature types to get ideas for what you might want to implement. Your new feature should help you improve the performance beyond what you got in Q4 if you wish to get these bonus points.

**Submit to the class leaderboard**. We also provide some test images for which you do not have labels. Apply your classifier on these test images and upload your classifications using the script `blah.py`. This will submit your entry to a leaderboard.

### Q6: Cool Bonus: Do something extra! (+10 points)

Implement, investigate or analyze something extra surrounding the topics in this assignment, and using the code you developed. For example, is there some other interesting question we could have asked? Is there any insightful visualization you can plot? Or maybe you can experiment with a spin on the loss function? If you try out something cool we'll give you points and might feature your results in the lecture.
