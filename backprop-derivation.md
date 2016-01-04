---
layout: page
permalink: /backprop-derivation/
---

Table of Contents:

- [Introduction](#intro)
- [Notations](#notation)
- [Building the circuit](#circuit)
- [Recursive computation of gradient](#gradrec)
- [Recursion base case](#recbase)
- [Summary](#summary)

<a name='intro'></a>
### Introduction

An intuitive understanding of neural networks and backpropagation can be gained by considering ANNs to be real-valued circuits of simple gates.
In the forward phase, each gate receives some inputs, performs an operation, and generates an output. In the backward phase, the gates receive the
gradient with respect to their output, and propagate it back to their inputs. This propagation takes the simple form of a product of the local gradient
and the received gradient. In this section we will use this view to develop the backpropagation equations for a fully connected neural network. The final
goal is to get expressions for the gradient of the cost function with respect to the weights and biases of a neural network.

<a name='notation'></a>
### Notations

\\(w^l\_{jk}\\) represents the weight from unit \\(k\\) of level \\(l-1\\) to unit \\(j\\) of level \\(l\\). \\(w^l\_{.k}\\) (note the dot before \\(k\\))
represents the vector of all weights to level \\(l\\) from unit \\(k\\) of level \\(l-1\\). \\(b^l\_j\\) is the bias of unit \\(j\\) of level
\\(l\\). The activation function of the hidden units is
represented by \\(\sigma\\). Note that we do not make any assumptions about the form of the activation function (sigmoid, tanh etc.). It is however
assumed that we can compute its gradient (or subgradient). \\(a^l\_j\\) is the activation of unit \\(j\\) of level \\(l\\). And finally, \\(z^l\_j\\)
is the biased weighted sum of inputs i.e. \\(z^l\_j=\sum\_k{w^l\_{jk}a^{l-1}\_k}+b^l\_j\\). Clearly, \\(a^l\_j=\sigma(z^l\_j)\\). We do not specify
the number of units in the hidden layers. Sums over quantities in a layer are over all units in the layer.

<a name='circuit'></a>
### Building the circuit

The neural network in consideration is a fully connected network with \\(L\\) layers. Let us start by designing the circuit that represents this network. We will focus on a
particular unit \\(j\\) of level \\(l\\). Each unit just computes a weighted sum of its inputs, adds a bias and applies the activation function.
\\(a^l\_j=\sigma(\sum\_k{w^l\_{jk}a^{l-1}\_k}+b^l\_j)\\). The atomic operations here are just multiplication, addition, and application of the
activation function; so we can represent a unit using gates corresponding to these operations as such:

<div class="fig figleft fighighlight">
  <svg width="799" height="250">
    <g>
      <defs>
        <marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto">
          <path d="M0,0 V4 L6,2 Z">
        </marker>
      </defs>

      <foreignObject x="25" y="10" width="10" height="10"><div style="font-size: 16px">\(w^l_{jk}\)</div></foreignObject>
      <line x1="50" y1="30" x2="150" y2="30" stroke="black" stroke-width="1"></line>
      <foreignObject x="80" y="0" width="10" height="10"><div style="color: green; font-size: 16px">\(w^l_{jk}\)</div></foreignObject>
      <line x1="150" y1="30" x2="175" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>

      <circle cx="30" cy="190" fill="white" stroke="black" stroke-width="1" r="20"></circle>
      <foreignObject x="15" y="175" width="10" height="10"><div style="font-size: 15px">\(\sigma^{l-1}\)</div></foreignObject>
      <line x1="50" y1="190" x2="75" y2="190" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <line x1="75" y1="190" x2="150" y2="190" stroke="black" stroke-width="1"></line>
      <line x1="150" y1="190" x2="175" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="80" y="160" width="10" height="10"><div style="color: green; font-size: 16px">\(a^{l-1}_k\)</div></foreignObject>

      <circle cx="195" cy="110" fill="white" stroke="black" stroke-width="1" r="20"></circle>
      <foreignObject x="190" y="98" width="10" height="10"><div style="font-size: 15px">\(*\)</div></foreignObject>
      <line x1="215" y1="110" x2="240" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <line x1="240" y1="110" x2="340" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="245" y="80" width="10" height="10"><div style="color: green; font-size: 16px">\(w^l_{jk}a^{l-1}_k\)</div></foreignObject>

      <line x1="315" y1="30" x2="340" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="300" y="10" width="10" height="10"><div style="font-size: 18px">\(\vdots\)</div></foreignObject>

      <line x1="315" y1="190" x2="340" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="300" y="170" width="10" height="10"><div style="font-size: 18px">\(\vdots\)</div></foreignObject>

      <foreignObject x="370" y="10" width="10" height="10"><div style="font-size: 16px">\(b^l_j\)</div></foreignObject>
      <line x1="385" y1="30" x2="485" y2="30" stroke="black" stroke-width="1"></line>
      <foreignObject x="415" y="0" width="10" height="10"><div style="color: green; font-size: 16px">\(b^l_j\)</div></foreignObject>
      <line x1="485" y1="30" x2="505" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>

      <circle cx="360" cy="110" fill="white" stroke="black" stroke-width="1" r="20"></circle>
      <foreignObject x="355" y="98" width="10" height="10"><div style="font-size: 15px">\(+\)</div></foreignObject>
      <line x1="380" y1="110" x2="405" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <line x1="405" y1="110" x2="505" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="400" y="80" width="10" height="10"><div style="color: green; font-size: 16px">\(\sum_kw^l_{jk}a^{l-1}_k\)</div></foreignObject>

      <circle cx="525" cy="110" fill="white" stroke="black" stroke-width="1" r="20"></circle>
      <foreignObject x="520" y="98" width="10" height="10"><div style="font-size: 15px">\(+\)</div></foreignObject>
      <line x1="545" y1="110" x2="570" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <line x1="570" y1="110" x2="620" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="575" y="80" width="10" height="10"><div style="color: green; font-size: 16px">\(z^l_j\)</div></foreignObject>

      <circle cx="640" cy="110" fill="white" stroke="black" stroke-width="1" r="20"></circle>
      <foreignObject x="635" y="95" width="10" height="10"><div style="font-size: 15px">\(\sigma^l\)</div></foreignObject>
      <line x1="660" y1="110" x2="685" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <line x1="685" y1="110" x2="785" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="685" y="80" width="10" height="10"><div style="color: green; font-size: 16px">\(a^l_j=\sigma(z^l_j)\)</div></foreignObject>
    </g>
  </svg>
  <div class="figcaption">
    Circuit representing a single unit in the neural network. Gate outputs are shown above the connecting "wires" in green. The activation gates are
    super scripted with their corresponding levels so they can be referred to unambiguously.
  </div>
  <div style="clear:both;"></div>
</div>

Let us analyze the above circuit. The whole circuit corresponds to a single perceptron unit in the conventional model of a neural network.
The multiply gate multiplies a single activation from the previous level with a corresponding weight. All such weighted
activations are summed up by the first add gate. The second add gate biases this weighted sum to produce \\(z^l\_j\\).
Finally the activation gate applies the activation function on this to produce \\(a^l\_j\\) which will act as input to the next level.

<a name='gradrec'></a>
### Recursive computation of gradient

Since gates behave locally, ignorant of everything they are not connected to, we can compute gradients across a chain through backpropagation provided
we know the gradient with respect to the final output. However, in the circuit we just constructed, we don't know any gradients to use as a starting point
and we seem to be stuck in a loop. Let us break this loop arbitrarily by giving a name to the gradient with respect to \\(z^l\_j\\): \\(\delta^l\_j\\). If
the final cost for the neural network circuit is \\(C\\), then \\(\delta^l\_j=\frac{\partial{C}}{\partial{z}^l\_j}\\). We will eventually need to find
a way of computing \\(\delta^l\_j\\), but for now let us use it to compute the other gradients.

Recall that the add gate simply distributes the gradient of its output to all its inputs. So \\(\delta^l\_j\\) passes unchanged to \\(b^l\_j\\),
\\(w^l\_{jk}a^{l-1}\_k\\) and we have \\(\frac{\partial{C}}{\partial{b^l\_j}}=\delta^l\_j\\), \\(\frac{\partial{C}}{\partial{w^l\_{jk}a^{l-1}\_k}}=
\delta^l\_j\\). The multiply gate switches its inputs and multiplies them with the output gradient. It sends \\(\delta^l\_ja^{l-1}\_k\\) back to
\\(w^l\_{jk}\\) which gives us \\(\frac{\partial{C}}{\partial{w^l\_{jk}}}=\delta^l\_ja^{l-1}\_k\\). To the \\(\sigma^{l-1}\\) gate it sends back
\\(\delta^l\_jw^l\_{jk}\\). Now the activation gate output is connected to all the multiply gates of the next level. It receives
gradients (as we've just calculated) from all the multiply gates and adds them up giving \\(\frac{\partial{C}}{\partial{a^{l-1}\_k}}=
\sum_j\delta^l\_jw^l\_{jk}\\). This can be written concisely in the form of a dot product: \\((\delta^l)^Tw^l\_{.k}\\) where \\(\delta^l\\) is a
vector of all the \\(\delta^l\_j\\)s. By the same logic, we can
compute the gradient with respect to \\(a^l\_j\\) as \\((\delta^{l+1})^Tw^{l+1}\_{.j}\\). These values are shown in the figure below.

<div class="fig figleft fighighlight">
  <svg width="799" height="250">
    <g>
      <defs>
        <marker id="arrowhead" refX="6" refY="2" markerWidth="6" markerHeight="4" orient="auto">
          <path d="M0,0 V4 L6,2 Z">
        </marker>
      </defs>

      <foreignObject x="25" y="10" width="10" height="10"><div style="font-size: 16px">\(w^l_{jk}\)</div></foreignObject>
      <line x1="50" y1="30" x2="150" y2="30" stroke="black" stroke-width="1"></line>
      <foreignObject x="80" y="0" width="10" height="10"><div style="color: green; font-size: 16px">\(w^l_{jk}\)</div></foreignObject>
      <foreignObject x="70" y="35" width="10" height="10"><div style="color: red; font-size: 16px">\(\delta^l_ja^{l-1}_k\)</div></foreignObject>
      <line x1="150" y1="30" x2="175" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>

      <circle cx="30" cy="190" fill="white" stroke="black" stroke-width="1" r="20"></circle>
      <foreignObject x="15" y="175" width="10" height="10"><div style="font-size: 15px">\(\sigma^{l-1}\)</div></foreignObject>
      <line x1="50" y1="190" x2="75" y2="190" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <line x1="75" y1="190" x2="150" y2="190" stroke="black" stroke-width="1"></line>
      <line x1="150" y1="190" x2="175" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="80" y="160" width="10" height="10"><div style="color: green; font-size: 16px">\(a^{l-1}_k\)</div></foreignObject>
      <foreignObject x="70" y="195" width="10" height="10"><div style="color: red; font-size: 16px">\(\delta^l_jw^l_{jk}\)</div></foreignObject>

      <circle cx="195" cy="110" fill="white" stroke="black" stroke-width="1" r="20"></circle>
      <foreignObject x="190" y="98" width="10" height="10"><div style="font-size: 15px">\(*\)</div></foreignObject>
      <line x1="215" y1="110" x2="240" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <line x1="240" y1="110" x2="340" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="245" y="80" width="10" height="10"><div style="color: green; font-size: 16px">\(w^l_{jk}a^{l-1}_k\)</div></foreignObject>
      <foreignObject x="255" y="115" width="10" height="10"><div style="color: red; font-size: 16px">\(\delta^l_j\)</div></foreignObject>

      <line x1="315" y1="30" x2="340" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="300" y="10" width="10" height="10"><div style="font-size: 18px">\(\vdots\)</div></foreignObject>

      <line x1="315" y1="190" x2="340" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="300" y="170" width="10" height="10"><div style="font-size: 18px">\(\vdots\)</div></foreignObject>

      <foreignObject x="370" y="10" width="10" height="10"><div style="font-size: 16px">\(b^l_j\)</div></foreignObject>
      <line x1="385" y1="30" x2="485" y2="30" stroke="black" stroke-width="1"></line>
      <foreignObject x="415" y="0" width="10" height="10"><div style="color: green; font-size: 16px">\(b^l_j\)</div></foreignObject>
      <foreignObject x="415" y="35" width="10" height="10"><div style="color: red; font-size: 16px">\(\delta^l_j\)</div></foreignObject>
      <line x1="485" y1="30" x2="505" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>

      <circle cx="360" cy="110" fill="white" stroke="black" stroke-width="1" r="20"></circle>
      <foreignObject x="355" y="98" width="10" height="10"><div style="font-size: 15px">\(+\)</div></foreignObject>
      <line x1="380" y1="110" x2="405" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <line x1="405" y1="110" x2="505" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="400" y="80" width="10" height="10"><div style="color: green; font-size: 16px">\(\sum_kw^l_{jk}a^{l-1}_k\)</div></foreignObject>
      <foreignObject x="420" y="115" width="10" height="10"><div style="color: red; font-size: 16px">\(\delta^l_j\)</div></foreignObject>

      <circle cx="525" cy="110" fill="white" stroke="black" stroke-width="1" r="20"></circle>
      <foreignObject x="520" y="98" width="10" height="10"><div style="font-size: 15px">\(+\)</div></foreignObject>
      <line x1="545" y1="110" x2="570" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <line x1="570" y1="110" x2="620" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="575" y="80" width="10" height="10"><div style="color: green; font-size: 16px">\(z^l_j\)</div></foreignObject>
      <foreignObject x="575" y="115" width="10" height="10"><div style="color: red; font-size: 16px">\(\delta^l_j\)</div></foreignObject>

      <circle cx="640" cy="110" fill="white" stroke="black" stroke-width="1" r="20"></circle>
      <foreignObject x="635" y="95" width="10" height="10"><div style="font-size: 15px">\(\sigma^l\)</div></foreignObject>
      <line x1="660" y1="110" x2="685" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <line x1="685" y1="110" x2="785" y2="110" stroke="black" stroke-width="1" marker-end="url(#arrowhead)"></line>
      <foreignObject x="685" y="80" width="10" height="10"><div style="color: green; font-size: 16px">\(a^l_j=\sigma(z^l_j)\)</div></foreignObject>
      <foreignObject x="680" y="115" width="10" height="10"><div style="color: red; font-size: 16px">\((\delta^{l+1})^Tw^{l+1}_{.j}\)</div></foreignObject>
    </g>
  </svg>
  <div class="figcaption">
    Gradients (shown below the wires in red) corresponding to a unit computed in terms of \(\delta^l_j\). For \(\sigma^{l-1}\), only the gradient received
    from the shown multiply gate is indicated.
  </div>
  <div style="clear:both;"></div>
</div>

Looking at the circuit, we can see that \\(\delta^l\_j\\) can be expressed using the gradient with respect to \\(a^l\_j\\). Specifically, we note that the
\\(\sigma^l\\) gate computes the local gradient as \\(\sigma'(z^l\_j)\\). It multiplies this with the gradient of its output \\((\delta^{l+1})^T
w^{l+1}\_{.j}\\) and passes it backwards. So \\(\delta^l\_j=\sigma'(z^l\_j)(\delta^{l+1})^Tw^{l+1}\_{.j}\\). Note that this depends only on values from the
next level in the network allowing us to compute \\(\delta^l\_j\\) recursively, and based on it, the other gradients corresponding to a unit.

<a name='recbase'></a>

### Recursion base case

To complete our backpropagation algorithm, we need to solve the base case of the recursion. First, we need to find an expression for
\\(\delta^L\_j\\) (\\(L\\) is the number of layers in the network). We will stick to the convention of not applying the activation
function on the final layer. So the output from the neural network is the vector of \\(z^L\_j\\)s which can be used to compute a cost \\(C\\)
(like cross-entropy or multiclass SVM loss). Since \\(\delta^L\_j=\frac{\partial{C}}{\partial{z^L\_j}}\\), this is readily computed as the form of
\\(C\\) is known. Using the \\(\delta^L\_j\\)s  as the starting point, all the \\(\delta^l\_j\\)s can be computed recursively.

The gradient with respect to the weights, \\(\delta^l\_ja^{l-1}\_k\\) depends on the activation from the previous level. So we need to
consider the special case when \\(l=1\\) i.e. the first hidden layer. However, this "issue" is remedied quite simply by adopting the convention that \\(a^0\\)
corresponds to the input vector \\(x\\) of the neural network. The correctness follows readily from the local nature of the gates: it does not matter
if the inputs are constants or outputs of other gates.

<a name='summary'></a>

### Summary

We have derived the gradient of a neural network cost function with respect to its weights and biases. The gradient takes the form of recursive
equations which can be evaluated by a forward and backward pass over the network. The equations are summarized here:

$$
\delta^L\_{j} = \frac{\partial{C}}{\partial{z^L\_j}}\\\\
\delta^l\_{j} = \sigma'(z^l\_j)(\delta^{l+1})^Tw^{l+1}\_{.j} \text{ for $l=1 \dots L-1$}\\\\
\frac{\partial{C}}{\partial{w^l\_{jk}}} = \delta^l\_ja^{l-1}\_k \text{ for $l=1 \dots L$}\\\\
\frac{\partial{C}}{\partial{b^l\_j}} = \delta^l\_j \text{ for $l=1 \dots L$}
$$
