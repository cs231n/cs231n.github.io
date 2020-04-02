---
layout: page
title: Jupyter Notebook Tutorial
permalink: /jupyter-tutorial/
---

In this class, we will use [Jupyter notebooks](https://jupyter.org/) for the programming assignments.
A Jupyter notebook lets you write and execute Python code in your web browser.
Jupyter notebooks make it very easy to tinker with code and execute it in bits
and pieces; for this reason they are widely used in scientific
computing.

**Note**: If your virtual environment was installed correctly (as per the [setup instructions]({{site.baseurl}}/setup-instructions/)), `jupyter notebook` should have been automatically installed. Just make sure your environment has been activate before proceeding.

To start a Jupyter notebook, run the following in your terminal:

```
jupyter notebook
```

Once your notebook server is up and running, point your web browser to http://localhost:8888 to
start using your notebooks. If everything worked correctly, you should
see a screen like this, showing all available notebooks in the current
directory:

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/file-browser.png'>
</div>

If you click through to a notebook file, you will see a screen like this:

<div class='fig figcenter'>
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
be enough to get you up and running on the assignments for this course.
