---
layout: page
title: IPython Tutorial
permalink: /ipython-tutorial/
---

In this class, we will use [IPython notebooks](http://ipython.org/) for the
programming assignments. An IPython notebook lets you write and execute Python
code in your web browser. IPython notebooks make it very easy to tinker with
code and execute it in bits and pieces; for this reason IPython notebooks are
widely used in scientific computing.

Installing and running IPython is easy. From the command line, the following
will install IPython:

```
pip install "ipython[notebook]"
```

Once you have IPython installed, start it with this command:

```
ipython notebook
```

Once IPython is running, point your web browser at http://localhost:8888 to
start using IPython notebooks. If everything worked correctly, you should
see a screen like this, showing all available IPython notebooks in the current
directory:

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/file-browser.png'>
</div>

If you click through to a notebook file, you will see a screen like this:

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-1.png'>
</div>

An IPython notebook is made up of a number of **cells**. Each cell can contain
Python code. You can execute a cell by clicking on it and pressing `Shift-Enter`.
When you do so, the code in the cell will run, and the output of the cell
will be displayed beneath the cell. For example, after running the first cell
the notebook looks like this:

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-2.png'>
</div>

Global variables are shared between cells. Executing the second cell thus gives
the following result:

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-3.png'>
</div>

By convention, IPython notebooks are expected to be run from top to bottom.
Failing to execute some cells or executing cells out of order can result in
errors:

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/notebook-error.png'>
</div>

After you have modified an IPython notebook for one of the assignments by
modifying or executing some of its cells, remember to **save your changes!**

<div class='fig figcenter'>
  <img src='/assets/ipython-tutorial/save-notebook.png'>
</div>

This has only been a brief introduction to IPython notebooks, but it should
be enough to get you up and running on the assignments for this course.
