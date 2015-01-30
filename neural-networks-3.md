---
layout: page
permalink: /neural-networks-3/
---

Table of Contents
(coming soon. This page is in draft form and is being updated)

## Learning

### Gradient Checks

In theory, performing a gradient check is as simple as comparing the analytic gradient to the numerical gradient. In practice, the process is much more involved and error prone. Here are some tips, tricks, and issues to watch out for:

**Use the centered formula**. The formula you may have seen for the finite difference approximation when evaluating the numerical gradient looks as follows:

$$
\frac{df(x)}{dx} = \frac{f(x + h) - f(x)}{h} \hspace{0.1in} \text{(bad, do not use)}
$$

where \\(h\\) is a very small number, in practice approximately 1e-5 or so. In practice, it turns out that it is much better to use the *centered* difference formula of the form:

$$
\frac{df(x)}{dx} = \frac{f(x + h) - f(x - h)}{2h} \hspace{0.1in} \text{(use instead)}
$$

This requires you to evaluate the loss function twice to check every single dimension of the gradient (so it is about 2 times as expensive), but the gradient approximation turns out to be much more precise. To see this, you can use Taylor expansion of \\(f(x+h)\\) and \\(f(x-h)\\) and verify that the first formula has an error on order of \\(O(h)\\), while the second formula only has error terms on order of \\(O(h^2)\\) (i.e. it is a second order approximation).

**Use relative error for the comparison**. What are the details of comparing the numerical gradient \\(f'\_n\\) and analytic gradient \\(f'\_a\\)? That is, how do we know if the two are not compatible? You might be temped to keep track of the difference \\(\mid f'\_a - f'\_n \mid \\) or its square and define the gradient check as failed if that difference is above a threshold. However, this is problematic. For example, consider the case where their difference is 1e-4. This seems like a very appropriate difference if the two gradients are about 1.0, so we'd consider the two gradients to match. But if the gradients were both on order of 1e-5 or lower, then we'd consider 1e-4 to be a huge difference and likely a failure. Hence, it is always more appropriate to consider the *relative error*:

$$
\frac{\mid f'\_a - f'\_n \mid}{\mid f'\_a \mid + \mid f'\_n \mid}
$$

which considers ther ratio of the differences to the ratio of the absolute values of both gradients. Notice that normally the relative error formula only includes one of the two terms (either one), but I prefer to add both to make it symmetric and to prevent underflow in the case where one of the two is zero. However, one must explicitly keep track of the case where both are zero (this can often be the case with ReLUs for example) and pass the gradient check in that edge case. In practice:

- relative error > 1e-2 usually means the gradient is probably wrong
- 1e-2 > relative error > 1e-4 should make you feel uncomfortable
- 1e-4 > relative error is usually okay for objectives with kinks. But if there are no kinks (e.g. use of tanh nonlinearities and softmax), then 1e-4 is too high.
- 1e-7 and less you should be happy.

Also keep in mind that the deeper the network, the lower the relative errors will be. So if you are gradient checking the input data for a 10-layer network, a relative error of 1e-2 might be okay because the errors build up on the way. Conversely, an error of 1e-2 for a single differentiable function likely indicates incorrect gradient.

**Kinks in the objective**. One source of inaccuracy to be aware of during gradient checking is the problem of *kinks*. Kinks refer to non-differentiable parts of an objective function, introduced by functions such as ReLU (\\(max(0,x)\\)), or the SVM loss, Maxout neurons, etc. Consider gradient checking the ReLU function at \\(x = -1e6\\). Since \\(x < 0\\), the analytic gradient at this point is exactly zero. However, the numerical gradient would suddenly compute a non-zero gradient because \\(f(x+h)\\) might cross over the kink (e.g. if \\(h > 1e-6\\)) and introduce a non-zero contribution. You might think that this is a pathological case, but in fact this case can be very common. For example, an SVM for CIFAR-10 contains up to 450,000 \\(max(0,x)\\) terms because there are 50,000 examples and each example yields 9 terms to the objective. Moreover, a Neural Network with an SVM classifier will contain many more kinks due to ReLUs.

Note that it is possible to know if a kink was crossed in the evaluation of the loss. This can be done by keeping track of the identities of all "winners" in a function of form \\(max(x,y)\\); That is, was x or y higher during the forward pass. If the identity of at least one winner changes when evaluating \\(f(x+h)\\) and then \\(f(x-h)\\), then a kink was crossed and the numerical gradient will not be exact.

**Be careful with the step size h**. It is not necessarily the case that smaller is better, because when \\(h\\) is much smaller, you may start running into numerical precision problems. Sometimes when the gradient doesn't check, it is possible that you change \\(h\\) to be 1e-4 or 1e-6 and suddenly the gradient will be correct. This [wikipedia article](http://en.wikipedia.org/wiki/Numerical_differentiation) contains a chart that plots the value of **h** on the x-axis and the numerical gradient error on the y-axis.

**Gradcheck during a "characteristic" mode of operation**. It is important to realize that a gradient check is performed at a particular (and usually random), single point in the space of parameters. Even if the gradient check succeeds at that point, it is not immediately certain that the gradient is correctly implemented globally. Additionally, a random initialization might not be the most "characteristic" point in the space of parameters and may in fact introduce pathological situations where the gradient seems to be correctly implemented but isn't. For instance, an SVM with very small weight initialization will assign almost exactly zero scores to all datapoints and the gradients will exhibit a particular pattern across all datapoints. An incorrect implementation of the gradient could still produce this pattern and not generalize to a more characteristic mode of operation where some scores are larger than others. Therefore, our recommendation is to use a short **burn-in** time during which the network is allowed to learn, and perform the gradient check after the loss starts to go down instead of performing the gradient check a single time at the first iteration. This could introduce pathological edge cases and mask an incorrect implementation of the gradient.

**Don't let the regularization overwhelm the data**. It is often the case that a loss function is a sum of the data loss and the regularization loss (e.g. L2 penalty on weights). One danger to be aware of is that the regularization loss may overwhelm the data loss, in which case the gradients will be primarily coming from the regularization term (which usually has a much simpler gradient expression). This can mask an incorrect implementation of the data loss gradient. Therefore, it is recommended to turn off regularization and check the data loss alone first, and then the regularization term second and independently. One way to perform the latter is to hack the code to remove the data loss contribution. Another way is to increase the regularization strength so as to ensure that its effect is non-negligeable in the gradient check, and that an incorrect implementation would be spotted.

**Remember to turn off dropout/augmentations**. When performing gradient check, remember to turn off any non-deterministic effects in the network, such as dropout, random data augmentations, etc. Otherwise these can clearly introduce huge errors when estimating the numerical gradient. The downside of turning off these effects is that you wouldn't be gradient checking them (e.g. it might be that dropout isn't backpropagated correctly). Therefore, a better solution might be to force a particular random seed before evaluating both \\(f(x+h)\\) and \\(f(x-h)\\), and when evaluating the analytic gradient.

**Check only few dimensions**. In practice the gradients can have sizes of million parameters. In these cases it is only practical to check some of the dimensions of the gradient and assume that the others are correct. **Be careful**: One issue to be careful with is to make sure to gradient check a few dimensions for every separate parameter. In some applications, people combine the parameters into a single large parameter vector for convenience. In these cases, for example, the biases could only take up a tiny number of parameters from the whole vector, so it is important to not sample at random but to take this into account and check that all parameters receive the correct gradients.

**Use only few datapoints**. If your gradcheck for only ~2 or 3 datapoints then you will almost certainly gradcheck for an entire batch. Using very few datapoints makes your gradient check faster and more efficient. Additionally,  loss functions that contain kinks (e.g. due to use of ReLUs or SVM etc.) will have fewer kinks with fewer datapoints, so it is less likely for you to cross one when you perform the finite different approximation. (More on kinks below).

### Before learning: sanity checks Tips/Tricks

Here are a few sanity checks you might consider running before you plunge into expensive optimization:

- **Look for correct loss at chance performance.** Make sure you're getting the loss you expect when you inititalize with small parameters. It's best to first check the data loss alone (so set regularization strength to zero). For example, for CIFAR-10 with a Softmax classifier we would expect the initial loss to be 2.302, because we expect a diffuse probability of 0.1 for each class (since there are 10 classes), and Softmax loss is the negative log probability of the correct class so: -ln(0.1) = 2.302. For The Weston Watkins SVM, we expect all desired margins to be violated (since all scores are approximately zero), and hence expect a loss of 9 (since margin is 1 for each wrong class). If you're not seeing these losses there might be issue with initialization.
- As a second sanity check, increasing the regularization strength should increase the loss
- **Overfit a tiny subset of data**. Lastly and most importantly, before training on the full dataset try to train on a tiny portion (e.g. 20 examples) of your data and make sure you can achieve zero cost. For this experiment it's also best to set regularization to zero, otherwise this can prevent you from getting zero cost. Unless you pass this sanity check with a small dataset it is not worth proceeding to the full dataset.

### Babysitting the learning process

There are multiple useful quantities you should monitor during training of a neural network. These plots are the window into the training process and should be utilized to get intuitions about different hyperparameter settings and how they should be changed for more efficient learning. 

The x-axis of the plots below are always in units of epochs, which measure how many times every example has been seen during training in expectation (e.g. one epoch means that every example has been seen once). It is preferrable to track epochs rather than iterations since the number of iterations depends on the arbitrary setting of batch size.

#### Loss function

The first quantity that is useful to track during training is the loss, as it is evaluated on the individual batches during the forward pass. Below is a cartoon diagram showing the loss over time, and especially what the shape might tell you about the learning rate:

<div class="fig figcenter fighighlight">
  <img src="/assets/nn3/learningrates.jpeg" width="49%">
  <img src="/assets/nn3/loss.jpeg" width="49%">
  <div class="figcaption">
    <b>Left:</b> A cartoon depicting the effects of different learning rates. With low learning rates the improvements will be linear. With high learning rates they will start to look more exponential. Higher learning rates will decay the loss faster, but they get stuck at worse values of loss (green line). This is because there is too much "energy" in the optimization and the parameters are bouncing around chaotically, unable to settle in a nice spot in the optimization landscape. <b>Right:</b> An example of a typical loss function over time, while training a small network on CIFAR-10 dataset. This loss function looks reasonable (it might indicate a slightly too small learning rate based on its speed of decay, but it's hard to say), and also indicates that the batch size might be a little too low (since the cost is a little too noisy).
  </div>
</div>

The amount of "wiggle" in the loss is related to the batch size. When the batch size is 1, the wiggle will be relatively high. When the batch size is the full dataset, the wiggle will be minimal because every gradient update should be improving the loss function monotonically (unless the learning rate is set too high).

#### Train/Val accuracy

The second important quantity to track while training a classifier is the validation/training accuracy. This plot can give you valuable insights into the amount of overfitting in your model:

<div class="fig figleft fighighlight">
  <img src="/assets/nn3/accuracies.jpeg">
  <div class="figcaption">
    The gap between the training and validation accuracy indicates the amount of overfitting. Two possible cases are shown in the diagram on the left. The blue validation error curve shows very small validation accuracy compared to the training accuracy, indicating strong overfitting (note, it's possible for the validation accuracy to even start to go down after some point). When you see this in practice you probably want to increase regularization (stronger L2 weight penalty, more dropout, etc.) or collect more data. The other possible case is when the validation accuracy tracks the training accuracy fairly well. This case indicates that your model capacity is not high enough: make the model larger by increasing the number of parameters.
  </div>
  <div style="clear:both"></div>
</div>

#### Ratio of weights:updates

The last quantity you might want to track is the ratio of the update magnitudes to to the value magnitudes. Note: *updates*, not the raw gradients (e.g. in vanilla sgd this would be the gradient multiplied by the learning rate). You might want to evaluate and track this ratio for every set of parameters independently. A rough heuristic is that this ratio should be somewhere around 1e-3. If it is lower than this then the learning rate might be too low. If it is higher then the learning rate is likely too high. Below is an example figure:

<div class="fig figcenter fighighlight">
  <img src="/assets/nn3/values.jpeg" width="49%">
  <img src="/assets/nn3/updates.jpeg" width="49%">
  <div class="figcaption">
    Example of a cross-validated 2-layer neural network where the learning rate is set relatively well. We are looking at the matrix of weights W1, and plotting the min and the max across all weights on the left. On the right, we plot the min and max of the updates for the weights, during gradient descent. The updates are getting smaller due to learning rate decay used in this example. Note that the approximate range of the updates is roughly 0.0002 and the range of values is about 0.02. This gives us a ratio of 0.0002 / 0.02 = 1e-2, which is within a relatively healthy limit for a smaller network.
  </div>
</div>

Instead of tracking the min or the max, some people prefer to compute and track the norm of the gradients and their updates instead. These metrics are usually correlated and often give approximately the same results.

#### First-layer Visualizations

Lastly, when one is working with image pixels it can be helpful and satisfying to plot the first-layer features visually:

<div class="fig figcenter fighighlight">
  <img src="/assets/nn3/weights.jpeg" width="43%" style="margin-right:10px;">
  <img src="/assets/nn3/cnnweights.jpg" width="49%">
  <div class="figcaption">
    Examples of visualized weights for the first layer of a neural network. <b>Left</b>: Noisy features indicate could be a symptom: Unconverged network, improperly set learning rate, very low weight regularization penalty. <b>Right:</b> Nice, smooth, clean and diverse features are a good indication that the training is proceeding well.
  </div>
</div>

### Parameter updates

coming soon

### Hyperparameter optimization

coming soon

### Implementation Tips and Tricks

coming soon

## Evaluation

### Model Ensembles

coming soon

## Additional References

- [SGD](http://research.microsoft.com/pubs/192769/tricks-2012.pdf) tips and tricks from Leon Bottou
- [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) (pdf)
- [deeplearning.net tutorial](http://www.deeplearning.net/tutorial/mlp.html) with Theano
- [Bengio guide](http://arxiv.org/pdf/1206.5533v2.pdf)
