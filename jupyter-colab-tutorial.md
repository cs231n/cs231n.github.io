---
layout: page
title: Jupyter Notebook / Google Colab Tutorial
permalink: /jupyter-colab-tutorial/
---

A Jupyter notebook lets you write and execute
Python code *locally* in your web browser. Jupyter notebooks
make it very easy to tinker with code and execute it in bits
and pieces; for this reason they are widely used in scientific
computing.
Colab on the other hand is Google's flavor of
Jupyter notebooks that is particularly suited for machine
learning and data analysis and that runs entirely in the *cloud*.
Colab is basically Jupyter notebook on steroids: it's free, requires no setup,
comes preinstalled with many packages, is easy to share with the world,
and benefits from free access to hardware accelerators like GPUs and TPUs (with some caveats).

To get yourself familiar with Python and notebooks, we'll be running
a short tutorial as a standalone Jupyter or Colab notebook. If you wish
to use Colab, click the `Open in Colab` badge below.

<div>
  <a href="https://colab.research.google.com/github/cs231n/cs231n.github.io/blob/master/python-colab.ipynb" target="_blank">
    <img class="colab-badge" src="/assets/badges/colab-open.svg" alt="Colab Notebook"/>
  </a>
</div>

If you wish to run the notebook locally make sure your virtual environment was installed correctly (as per the [setup instructions]({{site.baseurl}}/setup-instructions/)), activate it, then run `pip install notebook` to install Jupyter notebook. Next, [open the notebook](https://raw.githubusercontent.com/cs231n/cs231n.github.io/master/jupyter-notebook-tutorial.ipynb) and download it to a directory of your choice by right-clicking on the page and selecting `Save Page As`. Then `cd` to that directory and run the following in your terminal:

```
jupyter notebook
```

Once your notebook server is up and running, point your web browser to `http://localhost:8888` to
start using your notebooks. If everything worked correctly, you should
see a screen like this, showing all available notebooks in the current
directory:

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/file-browser.png'>
</div>

Click `jupyter-notebook-tutorial.ipynb` and follow the instructions in the notebook. Enjoy!

<!-- If you click through to a notebook file, you will see a screen like this: -->

<!-- <div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-1.png'>
</div>

A Jupyter notebook is made up of a number of **cells**. Each cell can contain
Python code. You can execute a cell by clicking on it (the highlight color will
switch from blue to green) and pressing `Shift-Enter`.
When you do so, the code in the cell will run, and the output of the cell
will be displayed beneath the cell. For example, after running the first cell,
the notebook shoud look like this:

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-2.png'>
</div>

Global variables are shared between cells. Executing the second cell thus gives
the following result:

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-3.png'>
</div>

There are a few keyboard shortcuts you should be aware of to make your notebook
experience more pleasant. To escape cell editing, press `esc`. The highlight color
should switch back to blue. To place a cell below the current one, press `b`.
To place a cell above the current one, press `a`. Finally, to delete a cell, press `dd`.

You can restart a notebook and clear all cells by clicking `Kernel -> Restart & Clear Output`.

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-restart.png'>
</div>

By convention, Jupyter notebooks are expected to be run from top to bottom.
Failing to execute some cells or executing cells out of order can result in
errors. After restarting the notebook, try running the second cell directly:

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-error.png'>
</div>

After you have modified a Jupyter notebook for one of the assignments by
modifying or executing some of its cells, remember to **save your changes!**

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/save-notebook.png'>
</div>

This has only been a brief introduction to Jupyter notebooks, but it should
be enough to get you up and running on the assignments for this course. -->
